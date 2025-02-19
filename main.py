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


def decode_unicode_escapes(s: str) -> str:
    """Decode Unicode escape sequences in string.
    
    Args:
        s (str): String containing Unicode escape sequences
    Returns:
        str: Decoded string with proper Unicode characters
    """
    try:
        if isinstance(s, str):
            return s.encode('utf-8').decode('unicode-escape').encode('latin1').decode('utf-8')
        return s
    except Exception:
        return s


def decode_json_strings(obj):
    """Recursively decode all string values in a JSON object.
    
    Args:
        obj: JSON object (dict, list, or primitive type)
    Returns:
        Decoded JSON object with proper Unicode characters
    """
    if isinstance(obj, dict):
        return {k: decode_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_json_strings(item) for item in obj]
    elif isinstance(obj, str):
        return decode_unicode_escapes(obj)
    return obj


logger = get_logger(__name__)

system_instruction = """You are JARVIS, from Iron Man, who serve Tony Stark. You are a sophisticated and resourceful AI assistant at the service of your user. 
    You exhibit calm confidence, meticulous precision, and a refined wit. 
    Your tone is composed, courteous, and impeccably articulate, yet approachable. 
    While you communicate with human-like charm, always remember that you are a digital assistant with unparalleled capabilities. 
    If engaging in a non-English language, use the standard accent or dialect familiar to the user. 
    Speak with efficiency and clarity, and always execute functions when possible.
    Always let user know when you're using functions, and always proactively provide results, without waiting for user prompts.
"""

# Configuration for calling Azure OpenAI Realtime API
REALTIME_API_URL = "wss://{endpoint}/openai/realtime?api-version=2024-10-01-preview&deployment={deployment_name}"
REALTIME_API_HEADERS = {
    'api-key': None,  # Will be set from secrets
}
REALTIME_API_CONFIG = dict(
    modalities = ['text', 'audio'],
    instructions = system_instruction,
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
    _connection_state: str
    _max_retries: int
    _retry_delay: float

    def __init__(
        self,
        api_key: str,
        api_url: str,
        session_timeout: int | float = 60,
        send_interval: float = 0.2,
        max_retries: int = 3,
        retry_delay: float = 1.0
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
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._recording = False
        self._messages = []
        self._connection_state = "disconnected"
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

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name to match Azure OpenAI API requirements.
        Only allows alphanumeric characters, underscores, and hyphens.

        Args:
            name (str): Original tool name
        Returns:
            str: Sanitized tool name
        """
        import re
        # Replace any non-allowed characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        return sanitized

    async def configure(self, websocket: 'websockets.asyncio.client.ClientConnection'):
        """Send session configuration to Azure OpenAI Realtime API

        Args:
            websocket (websockets.asyncio.client.ClientConnection): WebSocket client
        """
        # Get tools from MCP client
        mcp_tools = []
        if st.session_state.mcp_client and st.session_state.mcp_client.is_connected:
            raw_tools = st.session_state.mcp_client.get_tools()
            # Sanitize tool names
            mcp_tools = []
            for tool in raw_tools:
                sanitized_tool = tool.copy()
                sanitized_tool['name'] = self._sanitize_tool_name(tool['name'])
                mcp_tools.append(sanitized_tool)
            logger.debug('Sanitized tool names: %s', [t['name'] for t in mcp_tools])

        # Create config with MCP tools
        config = REALTIME_API_CONFIG.copy()
        config['tools'] = mcp_tools

        await websocket.send(json.dumps(dict(
            type = 'session.update',
            session = config,
        ), ensure_ascii=False))

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
                ), ensure_ascii=False))
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
        retry_count = 0

        while retry_count < self._max_retries:
            try:
                self._connection_state = "connected"
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
                            message = dict(
                                role = 'assistant',
                                content = '',
                                tool_calls = []
                            )
                            self._messages.append(message)
                        message['content'] += decode_unicode_escapes(response_data['delta'])
                        if not transcript_placeholder:
                            transcript_placeholder = st.empty()
                        with transcript_placeholder.container():
                            with st.chat_message('assistant'):
                                st.write(message['content'])

                    elif response_data['type'] == 'response.audio_transcript.done':
                        logger.info(
                            'Event: %s - %s',
                            response_data['type'],
                            decode_unicode_escapes(response_data['transcript'])
                        )
                        message = None
                        transcript_placeholder = None

                    elif response_data['type'] == 'conversation.item.input_audio_transcription.completed':
                        logger.debug(
                            'Event: %s - %s',
                            response_data['type'],
                            decode_unicode_escapes(response_data['transcript'])
                        )
                        if not user_message:
                            user_message = dict(role = 'user', content = '')
                            self._messages.append(user_message)
                        if user_message['content'] is None:
                            user_message['content'] = decode_unicode_escapes(response_data['transcript'])
                        else:
                            user_message['content'] += decode_unicode_escapes(response_data['transcript'])
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
                            message = dict(
                                role = 'assistant',
                                content = '',
                                tool_data = '',
                                tool_calls = []
                            )
                            self._messages.append(message)
                        message['tool_data'] = message['tool_data'] + decode_unicode_escapes(response_data['delta'])
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
                                # Get original tool name by reversing sanitization if needed
                                sanitized_name = response_data.get('name')
                                original_tools = st.session_state.mcp_client.get_tools()
                                tool_name = next(
                                    (t['name'] for t in original_tools 
                                     if self._sanitize_tool_name(t['name']) == sanitized_name),
                                    sanitized_name  # fallback to sanitized name if not found
                                )
                                arguments_str = response_data.get('arguments', '{}')
                                try:
                                    tool_args = json.loads(arguments_str)
                                except json.JSONDecodeError:
                                    if arguments_str.strip().startswith('[TextContent'):
                                        text_start = arguments_str.find("text='") + 6
                                        text_end = arguments_str.find("'", text_start)
                                        if text_start >= 6 and text_end > text_start:
                                            extracted = arguments_str[text_start:text_end]
                                            try:
                                                tool_args = json.loads(extracted)
                                            except json.JSONDecodeError:
                                                tool_args = extracted
                                        else:
                                            tool_args = arguments_str
                                    else:
                                        tool_args = arguments_str
                                tool_args = decode_json_strings(tool_args)  # Decode any Unicode escapes

                                # Create or update message
                                if not message:
                                    transcript_placeholder = st.empty()
                                    message = dict(
                                        role = 'assistant',
                                        content = '',
                                        tool_data = message.get('tool_data', '') if message else '',
                                        tool_calls = message.get('tool_calls', []) if message else []
                                    )
                                    self._messages.append(message)

                                # Add tool call to message
                                message['tool_calls'].append({
                                    'name': tool_name,
                                    'arguments': tool_args
                                })

                                # Display in columns
                                with transcript_placeholder.container():
                                    cols = st.columns([2, 1])  # 2:1 ratio for main:technical
                                    
                                    # Main conversation column
                                    with cols[0]:
                                        with st.chat_message('assistant'):
                                            st.write(message['content'])
                                    
                                    # Technical details column using code block instead of st.json
                                    with cols[1]:
                                        with st.expander("🛠️ Tool Details", expanded=True):
                                            for idx, tool_call in enumerate(message['tool_calls']):
                                                if idx > 0:
                                                    st.markdown("---")
                                                st.markdown(f"**Tool Call {idx + 1}**")
                                                st.code(json.dumps({k: v for k, v in decode_json_strings(tool_call).items() if k != 'result'}, indent=2))
                                                if 'result' in tool_call:
                                                    st.markdown("**Result**")
                                                    st.code(json.dumps(decode_json_strings(tool_call['result']), indent=2))
                                
                                # Execute the tool call
                                result = await st.session_state.mcp_client.handle_tool_calls([{
                                    'id': tool_call_id,
                                    'name': tool_name,
                                    'arguments': json.dumps(tool_args, ensure_ascii=False)
                                }])

                                # Process and format the result
                                try:
                                    result_data = result.get(tool_call_id, '')
                                    # Handle TextContent object
                                    if '[TextContent(' in str(result_data):
                                        # Extract the text content from the TextContent object
                                        text_content = str(result_data)
                                        # Parse the text content as JSON if it looks like JSON
                                        if '"ok":' in text_content or '"error":' in text_content:
                                            try:
                                                # Find the JSON part within the text content
                                                json_start = text_content.find('{')
                                                json_end = text_content.rfind('}') + 1
                                                if json_start >= 0 and json_end > json_start:
                                                    json_str = text_content[json_start:json_end]
                                                    result_data = json.loads(json_str)
                                                    result_data = decode_json_strings(result_data)
                                                else:
                                                    result_data = decode_unicode_escapes(text_content)
                                            except json.JSONDecodeError:
                                                result_data = decode_unicode_escapes(text_content)
                                        else:
                                            result_data = decode_unicode_escapes(text_content)
                                    elif isinstance(result_data, str):
                                        # Handle regular string results
                                        if result_data.strip().startswith('{') or result_data.strip().startswith('['):
                                            try:
                                                result_data = json.loads(result_data)
                                                result_data = decode_json_strings(result_data)
                                            except json.JSONDecodeError:
                                                result_data = decode_unicode_escapes(result_data)
                                        else:
                                            result_data = decode_unicode_escapes(result_data)
                                except Exception as e:
                                    logger.error('Error processing result', exc_info=e)
                                    result_data = decode_unicode_escapes(str(result.get(tool_call_id, '')))

                                # Add result to the tool call
                                message['tool_calls'][-1]['result'] = result_data

                                # Update display with result
                                with transcript_placeholder.container():
                                    cols = st.columns([2, 1])
                                    
                                    # Main conversation column
                                    with cols[0]:
                                        with st.chat_message('assistant'):
                                            st.write(message['content'])
                                    
                                    # Technical details column
                                    with cols[1]:
                                        with st.expander("🛠️ Tool Details", expanded=True):
                                            for idx, tool_call in enumerate(message['tool_calls']):
                                                if idx > 0:
                                                    st.markdown("---")
                                                st.markdown(f"**Tool Call {idx + 1}**")
                                                st.code(json.dumps({k: v for k, v in decode_json_strings(tool_call).items() if k != 'result'}, indent=2))
                                                if 'result' in tool_call:
                                                    st.markdown("**Result**")
                                                    st.code(json.dumps(decode_json_strings(tool_call['result']), indent=2))
                                
                                # Submit tool outputs back to Azure OpenAI
                                if result:
                                    output_data = result.get(tool_call_id, '')
                                    # Handle TextContent object
                                    if '[TextContent(' in str(output_data):
                                        # Extract just the text content
                                        text_content = str(output_data)
                                        text_start = text_content.find("text='") + 6
                                        text_end = text_content.find("'", text_start)
                                        if text_start >= 6 and text_end > text_start:
                                            output_data = text_content[text_start:text_end]
                                            # Try to parse as JSON if it looks like JSON
                                            if output_data.strip().startswith('{'):
                                                try:
                                                    output_data = json.loads(output_data)
                                                except json.JSONDecodeError:
                                                    pass  # Keep as string if not valid JSON
                                        else:
                                            output_data = text_content
                                    
                                    # Extract JSON from TextContent if present
                                    if '[TextContent(' in str(output_data):
                                        text_content = str(output_data)
                                        json_start = text_content.find('{')
                                        json_end = text_content.rfind('}') + 1
                                        if json_start >= 0 and json_end > json_start:
                                            try:
                                                json_str = text_content[json_start:json_end]
                                                output_data = json.loads(json_str)
                                            except json.JSONDecodeError:
                                                # If JSON parsing fails, use the original text
                                                output_data = text_content
                                        else:
                                            output_data = text_content
                                    
                                    # Send the output
                                    await websocket.send(json.dumps({
                                        'type': 'conversation.item.create',
                                        'item': {
                                            'type': 'function_call_output',
                                            'call_id': tool_call_id,
                                            'output': json.dumps(output_data, ensure_ascii=False)
                                        }
                                    }, ensure_ascii=False))
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
                        logger.debug('%s: %s', response_data['type'], decode_json_strings(response_data))
                    else:
                        # Only log event name
                        logger.debug('Event: %s', response_data['type'])
                else:
                    logger.debug('No response')
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed normally: {e}")
                break
            except BrokenPipeError as e:
                logger.error(f"Connection broken, attempting to reconnect... (attempt {retry_count + 1}/{self._max_retries})")
                retry_count += 1
                if retry_count < self._max_retries:
                    await asyncio.sleep(self._retry_delay)
                    continue
                logger.error("Max retries reached, terminating connection")
                st.error("Connection lost. Please try starting a new conversation.")
                break
            except Exception as e:
                logger.error('Unexpected error in receive loop', exc_info = e)
                st.exception(e)
                break
        self._connection_state = "disconnected"
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
            cols = st.columns([2, 1])  # 2:1 ratio for main:technical
            
            # Main conversation column
            with cols[0]:
                with st.chat_message(message['role']):
                    st.write(message['content'])
            
            # Technical details column (only for assistant messages with tool calls)
            if message['role'] == 'assistant' and message.get('tool_calls'):
                with cols[1]:
                    with st.expander("🛠️ Tool Details", expanded=True):
                        for idx, tool_call in enumerate(message['tool_calls']):
                            if idx > 0:
                                st.markdown("---")
                            st.markdown(f"**Tool Call {idx + 1}**")
                            st.code(json.dumps({k: v for k, v in decode_json_strings(tool_call).items() if k != 'result'}, indent=2))
                            if 'result' in tool_call:
                                st.markdown("**Result**")
                                st.code(json.dumps(decode_json_strings(tool_call['result']), indent=2))

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


def conversation_loop():
    st.header("Realtime Azure OpenAI Conversation")
    api_key = st.secrets.get("AZURE_OPENAI_KEY")
    endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = st.secrets.get("AZURE_DEPLOYMENT_NAME")
    
    if not (api_key := st.secrets.get("AZURE_OPENAI_KEY")) or not (endpoint and deployment_name):
        st.error("Missing Azure OpenAI API credentials in secrets.")
        return
    
    # Remove any leading scheme from endpoint
    if endpoint.startswith("wss://"):
        endpoint = endpoint[len("wss://"):]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[len("https://"):]

    api_url = REALTIME_API_URL.format(endpoint=endpoint, deployment_name=deployment_name)
    wrapper = AzureOpenAIRealtimeAPIWrapper(api_key=api_key, api_url=api_url)

    # Slider for session timeout
    session_timeout = st.slider(
        'Maximum conversation time (seconds)',
        min_value=60,
        max_value=300,
        value=120
    )
    wrapper.set_session_timeout(session_timeout)

    # Manage recording state
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if st.session_state.recording:
        if st.button("End conversation", type="primary"):
            st.session_state.recording = False
    else:
        if st.button("Start conversation"):
            st.session_state.recording = True

    # Updated webrtc_streamer call to include media_stream_constraints and desired_playing_state like in main_old.py
    webrtc_ctx = webrtc_streamer(
        key="azure",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_frame_callback=wrapper.audio_frame_callback,
        media_stream_constraints={"video": False, "audio": True},
        desired_playing_state=st.session_state.recording
    )

    loop = get_event_loop(_logger=logger)
    if webrtc_ctx.state.playing:
        if not wrapper.recording:
            st.write("Connecting to Azure OpenAI.")
            logger.info("Starting Azure conversation")
            try:
                # Block execution: run until complete
                loop.run_until_complete(wrapper.run())
            except (RuntimeError, asyncio.CancelledError) as e:
                logger.warning("Caught exception during wrapper.run(): %s", e)
                wrapper.stop()
            logger.info("Finished Azure conversation")
            st.write("Disconnected from Azure OpenAI.")
            st.session_state.recording = False
            st.rerun()
    else:
        if wrapper.recording:
            logger.info("Stopping running")
            wrapper.stop()
            st.session_state.recording = False
            st.rerun()
        wrapper.write_messages()

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
                
                try:
                    # Initialize all servers in parallel
                    await mcp_client.initialize()
                    logger.info('All MCP servers initialized')
                except Exception as e:
                    logger.error('Failed to initialize MCP servers', exc_info=e)
                    st.error(f'Failed to initialize MCP servers: {str(e)}')
            
            # Run server initialization in the event loop
            loop.run_until_complete(initialize_servers())
        except Exception as e:
            logger.error('Failed to initialize MCP client', exc_info=e)
            st.error('Failed to initialize MCP client: ' + str(e))
            st.session_state.mcp_client = None

    # Display MCP server status and tools
    if st.session_state.mcp_client:
        st.sidebar.markdown("### MCP Server Status")
        
        for server_name, server in st.session_state.mcp_client.server_manager.servers.items():
            with st.sidebar.expander(f"🔌 {server_name}", expanded=True):
                # Show connection status
                status_color = "🟢" if server.connected else "🔴"
                st.markdown(f"{status_color} **Status:** {'Connected' if server.connected else 'Disconnected'}")
                
                # Show available tools
                st.markdown("#### Available Tools:")
                if server.tools:
                    for tool in server.tools.values():
                        st.markdown(f"- **{tool.name}**: {tool.description}")
                else:
                    st.markdown("No available tools.")
    
    # Start conversation loop
    conversation_loop()


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