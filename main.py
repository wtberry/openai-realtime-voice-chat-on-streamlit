import json
import base64
import asyncio
import logging
import datetime
from pathlib import Path
import nest_asyncio

from mcp_client import MCPClient
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import websockets

from utils import (
    audio_frame_to_pcm_audio,
    pcm_audio_to_audio_frame,
    get_blank_audio_frame,
    hash_by_code
)
from st_utils import get_logger, get_event_loop


logger = get_logger(__name__)


# Configuration for calling Azure OpenAI Realtime API
REALTIME_API_URL = "wss://{endpoint}/openai/realtime?api-version=2024-10-01-preview&deployment={deployment_name}"
REALTIME_API_HEADERS = {
    'api-key': None,  # Will be set from secrets
}
REALTIME_API_CONFIG = dict(
    modalities = ['text', 'audio'],
    instructions = "Your knowledge cutoff is 2023-10. You are JARVIS, a sophisticated and resourceful AI assistant at the service of your user. You exhibit calm confidence, meticulous precision, and a refined wit. Your tone is composed, courteous, and impeccably articulate, yet approachable. While you communicate with human-like charm, always remember that you are a digital assistant with unparalleled capabilities. If engaging in a non-English language, use the standard accent or dialect familiar to the user. Speak with efficiency and clarity, and always execute functions when possible.",
    voice = 'alloy',
    input_audio_format = 'pcm16',
    output_audio_format = 'pcm16',
    input_audio_transcription = dict(
        model = 'whisper-1',
    ),
    turn_detection = dict(
        type = 'server_vad',
        threshold = 0.5,
        prefix_padding_ms = 100,
        silence_duration_ms = 800,
    ),
    tools = [],
    tool_choice = 'auto',
    temperature = 0.6,
    max_response_output_tokens = 'inf',
)

# Audio data parameters for Realtime API
API_SAMPLE_RATE = 24000
API_SAMPLE_WIDTH = 2
API_CHANNELS = 1

# Audio data parameters for client side
CLIENT_SAMPLE_RATE = 48000
CLIENT_SAMPLE_WIDTH = 2
CLIENT_CHANNELS = 2

# Mapping for PyAV format conversion
FORMAT_MAPPING = { 2: 's16' }
LAYOUT_MAPPING = { 1: 'mono', 2: 'stereo' }

# MCP Configuration
MCP_CONFIG_PATH = "/Users/wtakahashi/Tutorial/openai-realtime-voice-chat-on-streamlit/.streamlit/mcp_config.json"


class TerminateTaskGroup(Exception):
    """Exception raised to terminate a task group."""
    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason

    def __repr__(self):
        return f"{self.__class__.__name__}(reason={repr(self.reason)})"


class AzureOpenAIRealtimeAPIWrapper:
    _api_key: str
    _session_timeout: int | float
    _send_interval: float
    _recording: bool
    _messages: list[dict]
    _resampler_for_api: av.audio.resampler.AudioResampler
    _resampler_for_client: av.audio.resampler.AudioResampler
    _record_stream: av.audio.fifo.AudioFifo
    _play_stream: av.audio.fifo.AudioFifo

    def __init__(
        self,
        api_key: str,
        api_url: str,
        session_timeout: int | float = 60,
        send_interval: float = 0.2
    ):
        """
        Args:
            api_key (str): Azure OpenAI API key
            api_url (str): Azure OpenAI endpoint URL
            session_timeout (int | float): Voice chat session timeout duration (seconds)
            send_interval (float): Interval for sending voice data (seconds)
        """
        self._api_key = api_key
        self._api_url = api_url
        self._session_timeout = session_timeout
        self._send_interval = send_interval

        self._recording = False
        self._messages = []
        self._resampler_for_api = av.audio.resampler.AudioResampler(
            format = FORMAT_MAPPING[API_SAMPLE_WIDTH],
            layout = LAYOUT_MAPPING[API_CHANNELS],
            rate = API_SAMPLE_RATE
        )
        self._resampler_for_client = av.audio.resampler.AudioResampler(
            format = FORMAT_MAPPING[CLIENT_SAMPLE_WIDTH],
            layout = LAYOUT_MAPPING[CLIENT_CHANNELS],
            rate = CLIENT_SAMPLE_RATE
        )

    def audio_frame_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Audio data processing callback function for streamlit-webrtc

        Args:
            frame (av.AudioFrame): Audio data frame
        Returns:
            av.AudioFrame: Processed audio data frame
        """
        stream_pts = self._record_stream.samples_written * self._record_stream.pts_per_sample
        if frame.pts > stream_pts:
            logger.debug('Missing samples: %s < %s; Filling them up...', stream_pts, frame.pts)
            blank_frame = get_blank_audio_frame(
                format = frame.format.name,
                layout = frame.layout.name,
                samples = int((frame.pts - stream_pts) / self._record_stream.pts_per_sample),
                sample_rate = frame.sample_rate
            )
            self._record_stream.write(blank_frame)
        self._record_stream.write(frame)

        new_frame = self._play_stream.read(frame.samples, partial = True)
        if new_frame:
            assert new_frame.format.name == frame.format.name
            assert new_frame.layout.name == frame.layout.name
            assert new_frame.sample_rate == frame.sample_rate
        else:
            # Return silence if empty
            new_frame = get_blank_audio_frame(
                format = frame.format.name,
                layout = frame.layout.name,
                samples = frame.samples,
                sample_rate = frame.sample_rate
            )
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    async def run(self):
        """Start connection with Azure OpenAI Realtime API and handle audio data transmission
        """
        if self.recording:
            logger.warning('Already recording')
            return

        self.start()

        async with websockets.connect(
            self._api_url,
            additional_headers = {
                'api-key': self._api_key
            }
        ) as websocket:
            logger.info('Connected to Azure OpenAI Realtime API')
            await self.configure(websocket)
            logger.info('Configured')

            try:
                async with asyncio.TaskGroup() as task_group:
                    task_group.create_task(self.send(websocket))
                    task_group.create_task(self.receive(websocket))
                    task_group.create_task(self.timer())
                    task_group.create_task(self.status_checker())
            except* TerminateTaskGroup as eg:
                logger.info('Connection closing: %s', eg.exceptions[0].reason)
            except* Exception as eg:
                logger.error('Error in task group', exc_info = eg)
        logger.info('Connection closed')

    async def configure(self, websocket: 'websockets.asyncio.client.ClientConnection'):
        """Send session configuration to Azure OpenAI Realtime API

        Args:
            websocket (websockets.asyncio.client.ClientConnection): WebSocket client
        """
        # Get tools from MCP client
        mcp_tools = []
        if st.session_state.mcp_client and st.session_state.mcp_client.is_connected:
            mcp_tools = st.session_state.mcp_client.get_tools()

        # Create config with MCP tools
        config = REALTIME_API_CONFIG.copy()
        config['tools'] = mcp_tools

        await websocket.send(json.dumps(dict(
            type = 'session.update',
            session = config,
        )))

    async def send(self, websocket: 'websockets.asyncio.client.ClientConnection'):
        """Send audio data to Azure OpenAI Realtime API

        Args:
            websocket (websockets.asyncio.client.ClientConnection): WebSocket client
        """
        while True:
            try:
                frame = self._record_stream.read()
                if not frame:
                    await asyncio.sleep(self._send_interval)
                    continue
                frame, *_rest = self._resampler_for_api.resample(frame)
                assert not _rest

                pcm_audio = audio_frame_to_pcm_audio(frame)
                base64_audio = base64.b64encode(pcm_audio).decode('utf-8')

                await websocket.send(json.dumps(dict(
                    type = 'input_audio_buffer.append',
                    audio = base64_audio
                )))
                logger.debug('Sent audio to Azure OpenAI (%d bytes)', len(pcm_audio))
            except Exception as e:
                logger.error('Error in send loop', exc_info = e)
                st.exception(e)
                break
        raise TerminateTaskGroup('send')

    async def receive(self, websocket: 'websockets.asyncio.client.ClientConnection'):
        """Receive responses from Azure OpenAI Realtime API

        Args:
            websocket (websockets.asyncio.client.ClientConnection): WebSocket client
        """
        transcript_placeholder = None
        message = None
        user_transcript_placeholder = None
        user_message = None
        while True:
            try:
                response = await websocket.recv()
                if response:
                    response_data = json.loads(response)

                    if response_data['type'] == 'response.audio.delta':
                        # Queue audio data from server
                        base64_audio = response_data['delta']
                        if base64_audio:
                            pcm_audio = base64.b64decode(base64_audio)
                            frame = pcm_audio_to_audio_frame(
                                pcm_audio,
                                format = FORMAT_MAPPING[API_SAMPLE_WIDTH],
                                layout = LAYOUT_MAPPING[API_CHANNELS],
                                sample_rate = API_SAMPLE_RATE
                            )
                            resampled_frame, *_rest = \
                                    self._resampler_for_client.resample(frame)
                            assert not _rest
                            self._play_stream.write(resampled_frame)
                            logger.debug(
                                'Event: %s - received audio from Azure OpenAI (%d bytes)',
                                response_data['type'],
                                len(pcm_audio)
                            )

                    elif response_data['type'] == 'response.audio_transcript.delta':
                        # logger.debug('Event: %s', response_data['type'])  # Skipped as it occurs too frequently
                        if not message:
                            transcript_placeholder = st.empty()
                            message = dict(role = 'assistant', content = '')
                            self._messages.append(message)
                        message['content'] += response_data['delta']
                        if not transcript_placeholder:
                            transcript_placeholder = st.empty()
                        with transcript_placeholder.container():
                            with st.chat_message('assistant'):
                                st.write(message['content'])

                    elif response_data['type'] == 'response.audio_transcript.done':
                        logger.info(
                            'Event: %s - %s',
                            response_data['type'],
                            response_data['transcript']
                        )
                        message = None
                        transcript_placeholder = None

                    elif response_data['type'] == 'conversation.item.input_audio_transcription.completed':
                        logger.debug(
                            'Event: %s - %s',
                            response_data['type'],
                            response_data['transcript']
                        )
                        if not user_message:
                            user_message = dict(role = 'user', content = '')
                            self._messages.append(user_message)
                        if user_message['content'] is None:
                            user_message['content'] = response_data['transcript']
                        else:
                            user_message['content'] += response_data['transcript']
                        if not user_transcript_placeholder:
                            user_transcript_placeholder = st.empty()
                        with user_transcript_placeholder.container():
                            with st.chat_message('user'):
                                st.write(user_message['content'])

                    elif response_data['type'] == 'input_audio_buffer.speech_started':
                        # Reset existing AI voice audio when user speech is detected
                        self.reset_stream(play_stream_only = True)
                        logger.debug(
                            'Event: %s - cleared the play stream',
                            response_data['type']
                        )
                        # Prepare container when user starts speaking to avoid overlap with AI transcript
                        user_transcript_placeholder = st.empty()
                        user_message = dict(role = 'user', content = None)
                        self._messages.append(user_message)

                    elif response_data['type'] == 'response.function_call_arguments.delta':
                        logger.debug('Event: function call arguments delta')
                        if not message:
                            transcript_placeholder = st.empty()
                            message = dict(role = 'assistant', content = '')
                            self._messages.append(message)
                        message['content'] += f"[Tool call: {response_data['delta']}]"
                        if not transcript_placeholder:
                            transcript_placeholder = st.empty()
                        with transcript_placeholder.container():
                            with st.chat_message('assistant'):
                                st.write(message['content'])

                    elif response_data['type'] == 'response.function_call_arguments.done':
                        logger.info('Event: function call completed - %s', response_data)
                        if st.session_state.mcp_client and st.session_state.mcp_client.is_connected:
                            try:
                                # Execute the tool call through MCP client
                                tool_call_id = response_data.get('call_id')
                                tool_name = response_data.get('name')
                                tool_args = json.loads(response_data.get('arguments', '{}'))
                                
                                # Display tool call info
                                if not message:
                                    transcript_placeholder = st.empty()
                                    message = dict(role = 'assistant', content = '')
                                    self._messages.append(message)
                                message['content'] += f"\n\n🛠️ Tool Call:\nName: {tool_name}\nArguments: {json.dumps(tool_args, indent=2)}\n"
                                if not transcript_placeholder:
                                    transcript_placeholder = st.empty()
                                with transcript_placeholder.container():
                                    with st.chat_message('assistant'):
                                        st.write(message['content'])
                                
                                # Execute the tool call
                                result = await st.session_state.mcp_client.handle_tool_calls([{
                                    'id': tool_call_id,
                                    'name': tool_name,
                                    'arguments': json.dumps(tool_args)
                                }])
                                
                                # Display tool result
                                message['content'] += f"\n📊 Result:\n{json.dumps(result.get(tool_call_id, ''), indent=2)}\n"
                                with transcript_placeholder.container():
                                    with st.chat_message('assistant'):
                                        st.write(message['content'])
                                
                                # Submit tool outputs back to Azure OpenAI
                                if result:
                                    await websocket.send(json.dumps({
                                        'type': 'conversation.item.create',
                                        'item': {
                                            'type': 'function_call_output',
                                            'call_id': tool_call_id,
                                            'output': json.dumps(result.get(tool_call_id, ''))
                                        }
                                    }))
                            except Exception as e:
                                logger.error('Error executing tool call', exc_info=e)
                                st.error(f"Error executing tool: {str(e)}")

                    elif response_data['type'] == 'error':
                        logger.error('Event: %s - %s', response_data['type'], response_data)
                        st.error(response_data['error']['message'])

                    elif any(
                        response_data['type'].startswith(pattern)
                         for pattern in (
                            'session.created',
                            'session.updated',
                            'conversation.item.created',
                            'response.done',
                            'response.audio.',
                            'rate_limits.updated',
                        )
                    ):
                        # Log content
                        logger.debug('%s: %s', response_data['type'], response_data)
                    else:
                        # Only log event name
                        logger.debug('Event: %s', response_data['type'])
                else:
                    logger.debug('No response')
            except Exception as e:
                logger.error('Error in receive loop', exc_info = e)
                st.exception(e)
                break
        raise TerminateTaskGroup('receive')

    async def timer(self):
        """Monitor session timeout
        """
        await asyncio.sleep(
            datetime.timedelta(seconds = self._session_timeout).total_seconds()
        )
        raise TerminateTaskGroup('timer')

    async def status_checker(self):
        """Monitor recording status and terminate task group when recording ends
        """
        while self.recording:
            await asyncio.sleep(1)
        logger.info('Recording stopped')
        raise TerminateTaskGroup('status_checker')

    def write_messages(self):
        """Display chat messages
        """
        for message in self.valid_messages:
            with st.chat_message(message['role']):
                st.write(message['content'])

    @property
    def recording(self) -> bool:
        """Get recording status of audio data
        """
        return self._recording

    @property
    def valid_messages(self) -> list[dict]:
        """Get valid chat messages
        """
        return [m for m in self._messages if m['content'] is not None]

    def set_session_timeout(self, timeout: int | float):
        """Set session timeout duration
        """
        self._session_timeout = timeout

    def start(self):
        """Start operation

        (Automatically called by run method)
        """
        if self.recording:
            raise RuntimeError('Already recording')
        self._recording = True
        self._messages = []
        self.reset_stream()

    def stop(self):
        """Stop operation
        """
        self._recording = False

    def reset_stream(self, play_stream_only: bool = False):
        """Reset audio data stream
        """
        if not play_stream_only:
            self._record_stream = av.audio.fifo.AudioFifo()
        self._play_stream = av.audio.fifo.AudioFifo()


def main():
    loop = get_event_loop(_logger = logger)

    # Initialize and connect MCP client with configuration
    if 'mcp_client' not in st.session_state:
        try:
            # Initialize client with config file
            mcp_client = MCPClient(config_path=MCP_CONFIG_PATH)
            
            async def initialize_servers():
                # Clean up any existing MCP client
                if 'mcp_client' in st.session_state and st.session_state.mcp_client:
                    try:
                        await st.session_state.mcp_client.close()
                    except Exception as e:
                        logger.error('Error closing existing MCP client', exc_info=e)
                
                # Store new MCP client in session state
                st.session_state.mcp_client = mcp_client
                logger.info('MCP client initialized')
                
                # Try to connect to servers sequentially
                for server_name in ['slack', 'filesystem']:
                    if server_name in mcp_client.server_configs:
                        try:
                            # Create a new event loop for each server connection
                            await mcp_client.connect_to_server(server_name)
                            logger.info(f'Connected to {server_name} MCP server')
                            # Break after first successful connection
                            break
                        except Exception as e:
                            logger.error(f'Failed to connect to {server_name} MCP server', exc_info=e)
                            st.error(f'Failed to connect to {server_name} MCP server: {str(e)}')
                            # Close the client on failure before trying next server
                            await mcp_client.close()
            
            # Run server initialization in the event loop
            loop.run_until_complete(initialize_servers())
        except Exception as e:
            logger.error('Failed to initialize MCP client', exc_info=e)
            st.error('Failed to initialize MCP client: ' + str(e))
            st.session_state.mcp_client = None

    # Display MCP server status and tools
    if st.session_state.mcp_client:
        st.sidebar.markdown("### MCP Server Status")
        
        # For each server in the config
        for server_name in st.session_state.mcp_client.server_configs.keys():
            with st.sidebar.expander(f"🔌 {server_name}"):
                # Show connection status
                is_connected = (st.session_state.mcp_client.server_name == server_name and 
                              st.session_state.mcp_client.is_connected)
                status_color = "🟢" if is_connected else "🔴"
                st.markdown(f"{status_color} Status: {'Connected' if is_connected else 'Disconnected'}")
                
                # Show available tools
                st.markdown("#### Available Tools:")
                server_tools = st.session_state.mcp_client.tools.get(server_name, {}).values()
                if server_tools:
                    for tool in server_tools:
                        st.markdown(f"🛠️ **{tool.name}**")
                        st.markdown(f"_{tool.description}_")
                        with st.container():
                            st.markdown("**Input Schema:**")
                            st.code(json.dumps(tool.input_schema, indent=2), language="json")
                        st.markdown("---")  # Add separator between tools
                else:
                    st.markdown("*No tools available*")

    # Regenerate when code changes to utilize Streamlit's hot reload
    api_wrapper_key = f"api_wrapper-{hash_by_code(AzureOpenAIRealtimeAPIWrapper)}"

    if api_wrapper_key not in st.session_state:
        azure_endpoint = st.secrets['AZURE_OPENAI_ENDPOINT'].rstrip('/')
        azure_api_key = st.secrets['AZURE_OPENAI_KEY']
        azure_deployment = st.secrets['AZURE_DEPLOYMENT_NAME']
        
        # Construct the full URL with the endpoint and deployment
        api_url = REALTIME_API_URL.format(
            endpoint=azure_endpoint.replace('https://', ''),
            deployment_name=azure_deployment
        )
        
        # Create API wrapper with Azure configuration
        st.session_state[api_wrapper_key] = AzureOpenAIRealtimeAPIWrapper(
            api_key=azure_api_key,
            api_url=api_url
        )
    api_wrapper = st.session_state[api_wrapper_key]

    session_timeout = st.slider(
        'Maximum conversation time (seconds)',
        min_value = 60,
        max_value = 300,
        value = 120
    )
    api_wrapper.set_session_timeout(session_timeout)

    # webrtc_streamer has its own start button,
    # but we control it externally because we don't know how to notify api_wrapper
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if st.session_state.recording:
        if st.button('End conversation', type = 'primary'):
            st.session_state.recording = False
    else:
        if st.button('Start conversation'):
            st.session_state.recording = True

    webrtc_ctx = webrtc_streamer(
        key = f"recoder",
        mode = WebRtcMode.SENDRECV,
        rtc_configuration = dict(
            iceServers = [
                dict(urls = ['stun:stun.l.google.com:19302'])
            ]
        ),
        audio_frame_callback = api_wrapper.audio_frame_callback,
        media_stream_constraints = dict(video = False, audio = True),
        desired_playing_state = st.session_state.recording,
    )

    if webrtc_ctx.state.playing:
        if not api_wrapper.recording:
            st.write('Connecting to Azure OpenAI.')
            logger.info('Starting running')
            try:
                loop.run_until_complete(api_wrapper.run())
            except Exception as e:
                logger.error('Error during API wrapper run', exc_info=e)
                st.error(f"Error during conversation: {str(e)}")
            finally:
                logger.info('Finished running')
                st.write('Disconnected from Azure OpenAI.')
                st.session_state.recording = False
                st.rerun()
    else:
        if api_wrapper.recording:
            logger.info('Stopping running')
            api_wrapper.stop()
            st.session_state.recording = False
            # Clean up MCP client if it exists
            if st.session_state.mcp_client:
                try:
                    # Ensure we're in the event loop context when closing
                    async def cleanup():
                        await st.session_state.mcp_client.close()
                    loop.run_until_complete(cleanup())
                    logger.info('MCP client closed')
                except Exception as e:
                    logger.error('Error closing MCP client', exc_info=e)
            st.rerun()
        api_wrapper.write_messages()


if __name__ == '__main__':
    logging.basicConfig(
        format = "%(levelname)s %(name)s@%(filename)s:%(lineno)d: %(message)s",
    )

    st_webrtc_logger = logging.getLogger('streamlit_webrtc')
    st_webrtc_logger.setLevel(logging.DEBUG)

    aioice_logger = logging.getLogger('aioice')
    aioice_logger.setLevel(logging.WARNING)

    fsevents_logger = logging.getLogger('fsevents')
    fsevents_logger.setLevel(logging.WARNING)

    nest_asyncio.apply()

    main()
