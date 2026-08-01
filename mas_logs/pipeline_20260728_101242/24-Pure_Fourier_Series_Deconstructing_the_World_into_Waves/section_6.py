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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Summary & Real-World Echoes", 
            [
                "Fourier Series bridges the time and frequency domains.",
                "Complex signals become simple lists of frequency components.",
                "These recipes power modern technology like digital audio compression."
            ]
        )
        
        # Colors for matching
        COLOR_1 = "#00FFFF"  # Cyan
        COLOR_2 = "#FFFF00"  # Yellow
        COLOR_3 = "#FFFFFF"  # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        # Create a complex time-domain wave
        time_axes = Axes(
            x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1],
            x_length=3, y_length=2,
            axis_config={"include_tip": False}
        ).set_color(GRAY)
        
        def wave_func(x):
            return 0.7 * np.sin(2 * PI * x) + 0.3 * np.sin(4 * PI * x) + 0.2 * np.sin(6 * PI * x)
        
        wave = time_axes.plot(wave_func, color=COLOR_1)
        wave_group = VGroup(time_axes, wave)
        self.place_in_area(wave_group, "A1", "C3", scale_factor=0.8)
        
        self.play(Create(time_axes), Create(wave))
        self.wait(1)

        # Create frequency bars for morphing
        bar_heights = [0.7, 0.3, 0.2, 0.1, 0.05]
        bars = VGroup(*[
            Rectangle(width=0.4, height=h, fill_opacity=0.8, fill_color=COLOR_1, stroke_color=WHITE)
            for h in bar_heights
        ]).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(bars, "A1", "C3", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(wave, bars),
            FadeOut(time_axes)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_2)
        
        # Transform bars to COLOR_2 and show a 'list' of values
        self.play(bars.animate.set_color(COLOR_2))
        
        freq_list = VGroup(*[
            MathTex(f"f_{i} = {h:.1f}", font_size=24, color=COLOR_2)
            for i, h in enumerate(bar_heights)
        ]).arrange(DOWN, aligned_edge=LEFT)
        
        self.place_in_area(freq_list, "A4", "C6", scale_factor=1.0)
        
        self.play(Write(freq_list))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_3)
        
        # Smartphone icon
        phone_body = RoundedRectangle(corner_radius=0.2, height=3, width=1.5, color=WHITE)
        screen = Rectangle(height=2.5, width=1.3, color=WHITE, fill_opacity=0.1)
        home_button = Circle(radius=0.1, color=WHITE)
        home_button.next_to(screen, DOWN, buff=0.1)
        smartphone = VGroup(phone_body, screen, home_button)
        
        # Fix for Issue 36: Adjusted position and scale
        self.place_in_area(smartphone, "D4", "F5", scale_factor=0.9)
        
        # Vibrating bars inside smartphone
        v_bars = VGroup(*[
            Rectangle(width=0.15, height=np.random.uniform(0.2, 1.2), fill_opacity=1, color=WHITE)
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.05)
        v_bars.move_to(screen.get_center())
        
        self.play(FadeIn(smartphone), FadeIn(v_bars))
        
        # Updater for vibrating effect
        def update_bars(m):
            for bar in m:
                new_h = np.random.uniform(0.1, 1.2)
                bar.stretch_to_fit_height(new_h)
                bar.move_to(screen.get_center(), aligned_edge=DOWN)
        
        v_bars.add_updater(update_bars)
        
        # Flash MP3 and JPEG
        mp3_text = Text("MP3", font_size=36, color=COLOR_1)
        jpeg_text = Text("JPEG", font_size=36, color=COLOR_1)
        
        # Fix for Issue 34: self.place_at_grid(mp3_text, 'D3', scale_factor=1.1)
        self.place_at_grid(mp3_text, "D3", scale_factor=1.1)
        # Fix for Issue 35: self.place_at_grid(jpeg_text, 'E3', scale_factor=1.1)
        self.place_at_grid(jpeg_text, "E3", scale_factor=1.1)
        
        self.play(Flash(mp3_text, color=COLOR_1, flash_radius=0.5), Write(mp3_text))
        self.wait(0.5)
        self.play(Flash(jpeg_text, color=COLOR_1, flash_radius=0.5), Write(jpeg_text))
        
        self.wait(2)
        v_bars.remove_updater(update_bars)
        
        # Cleanup
        self.play(
            FadeOut(bars),
            FadeOut(freq_list),
            FadeOut(smartphone),
            FadeOut(v_bars),
            FadeOut(mp3_text),
            FadeOut(jpeg_text)
        )
        self.wait(1)
