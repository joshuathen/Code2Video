from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "The Hook: The Whistle-Bot's Pure Tones"
        lecture_lines = [
            "- Meet Whistle-Bot, a master of pure periodic tones.",
            "- Most sounds are messy, but Whistle-Bot prefers smooth oscillations.",
            "- He moves in a jerky, jagged square wave pattern.",
            "- Can we recreate this jump using only smooth movements?",
            "- Let's explore the power of mathematical symmetry."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        BOT_COLOR = "#FFD700"
        SINE_COLOR = "#ADD8E6"
        HARMONIC_COLOR = "#90EE90"
        
        # Setup Axes in the grid area - Fix Issue 32 (B3 to E6)
        axes = Axes(
            x_range=[0, 4 * PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=4.0,
            y_length=3.0,
            tips=False,
            axis_config={"include_numbers": False, "color": GREY}
        )
        self.place_in_area(axes, "B3", "E6", scale_factor=0.9)
        
        # Whistle-Bot representation
        bot_body = Square(side_length=0.4, color=BOT_COLOR, fill_opacity=0.8)
        bot_head = Circle(radius=0.15, color=BOT_COLOR, fill_opacity=1).next_to(bot_body, UP, buff=0.05)
        bot_eye1 = Dot(radius=0.03, color=BLACK).move_to(bot_head.get_center() + 0.05 * LEFT + 0.02 * UP)
        bot_eye2 = Dot(radius=0.03, color=BLACK).move_to(bot_head.get_center() + 0.05 * RIGHT + 0.02 * UP)
        whistle_bot = VGroup(bot_body, bot_head, bot_eye1, bot_eye2)
        
        # ValueTracker for movement
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BOT_COLOR)
        # Fix Issue 31/33 (D4, scale 0.4)
        self.place_at_grid(whistle_bot, "D4", scale_factor=0.4)
        self.play(FadeIn(whistle_bot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(SINE_COLOR)
        self.play(Create(axes))
        
        sine_func = lambda x: np.sin(x)
        sine_wave = axes.plot(sine_func, color=SINE_COLOR)
        self.play(Create(sine_wave))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BOT_COLOR)
        
        # Square wave
        square_wave_func = lambda x: 1.0 if np.sin(x) >= 0 else -1.0
        square_wave = axes.plot(
            square_wave_func, 
            color=BOT_COLOR, 
            use_smoothing=False, 
            discontinuities=[PI, 2*PI, 3*PI]
        )
        
        # Add updater to make Whistle-Bot follow the square wave path
        def update_bot(m):
            t = time_tracker.get_value()
            y_val = square_wave_func(t)
            m.move_to(axes.c2p(t, y_val))
            
        whistle_bot.add_updater(update_bot)
        
        self.play(Create(square_wave))
        # Bot moves across the screen
        self.play(time_tracker.animate.set_value(4 * PI), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(SINE_COLOR)
        
        # First approximation term: (4/pi) * sin(x)
        fundamental_func = lambda x: (4/PI) * np.sin(x)
        fundamental_wave = axes.plot(fundamental_func, color=SINE_COLOR)
        
        # Transition from pure sine to the fundamental component
        self.play(FadeOut(sine_wave), Create(fundamental_wave))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(HARMONIC_COLOR)
        
        # Next term in series: (4/(3*pi)) * sin(3x)
        third_harmonic_func = lambda x: (4/(3*PI)) * np.sin(3*x)
        third_harmonic_wave = axes.plot(third_harmonic_func, color=HARMONIC_COLOR)
        
        # The sum of terms
        sum_func = lambda x: (4/PI) * (np.sin(x) + np.sin(3*x)/3)
        sum_wave = axes.plot(sum_func, color=WHITE)
        
        self.play(Create(third_harmonic_wave))
        self.wait(1)
        
        # Show how combining them starts to approximate the square wave
        self.play(
            FadeOut(fundamental_wave),
            FadeOut(third_harmonic_wave),
            Create(sum_wave)
        )
        self.wait(2)
        
        # Remove updaters to prevent unexpected behavior at end of scene
        whistle_bot.clear_updaters()
