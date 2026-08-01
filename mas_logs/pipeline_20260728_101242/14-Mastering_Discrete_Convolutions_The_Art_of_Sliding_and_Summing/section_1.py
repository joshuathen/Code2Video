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
        # Setup the scene with title and lecture lines
        self.setup_layout(
            "The Big Picture: Why Convolve?", 
            [
                "Convolution combines two signals into a single new output.",
                "It powers image filters, audio effects, and computer vision.",
                "A sharp image blurs by averaging pixels with neighbors."
            ]
        )
        
        # Colors for the section
        COLOR_SIGNAL_1 = BLUE_C
        COLOR_SIGNAL_2 = YELLOW_C
        COLOR_HIGHLIGHT_1 = "#00FFFF"
        COLOR_HIGHLIGHT_2 = "#FF00FF"
        
        # === Animation for Lecture Line 1 ===
        # Convolution combines two signals into a single new output.
        self.lecture[0].set_color(COLOR_SIGNAL_1)
        
        # Blue Pulse signal
        pulse = VGroup(
            Line(LEFT, ORIGIN),
            Line(ORIGIN, UP),
            Line(UP, RIGHT + UP),
            Line(RIGHT + UP, RIGHT),
            Line(RIGHT, 2 * RIGHT)
        ).set_color(COLOR_SIGNAL_1)
        self.place_in_area(pulse, 'A1', 'B3', scale_factor=0.6)
        
        # Yellow Bell Curve signal
        bell = FunctionGraph(
            lambda x: 2 * np.exp(-x**2),
            x_range=[-2, 2]
        ).set_color(COLOR_SIGNAL_2)
        self.place_in_area(bell, 'A4', 'B6', scale_factor=0.4)
        
        self.play(Create(pulse), Create(bell))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It powers image filters, audio effects, and computer vision.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT_1)
        
        # [Asset: ...image.svg]
        img_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/image.svg", color=WHITE)
        self.place_at_grid(img_icon, 'C2', scale_factor=0.5)
        
        # Audio Wave Icon
        audio_icon = VGroup(*[
            Line(0.5*DOWN, 0.5*UP).shift(x*0.15*RIGHT) 
            for x in range(-4, 5)
        ]).set_color(WHITE)
        # Give it a wave shape
        for i, line in enumerate(audio_icon):
            line.set_y_length(0.2 + 0.6 * np.abs(np.sin(i * 0.8)))
        # Issue 21: Change scale to 0.5
        self.place_at_grid(audio_icon, 'C5', scale_factor=0.5)
        
        # [Asset: ...eye.svg]
        eye_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color=WHITE)
        # Issue 22: Reposition to D4
        self.place_at_grid(eye_icon, 'D4', scale_factor=0.5)
        
        # Small Pixel Grid
        vision_grid = VGroup(*[
            Square(side_length=0.25, color=GRAY_D, stroke_width=1) 
            for _ in range(16)
        ]).arrange_in_grid(4, 4, buff=0.05)
        # Positioned near eye icon
        self.place_at_grid(vision_grid, 'E4', scale_factor=1.0)
        
        self.play(FadeIn(img_icon), FadeIn(audio_icon))
        # Highlight pulse effect
        self.play(
            img_icon.animate.set_color(COLOR_HIGHLIGHT_1),
            audio_icon.animate.set_color(COLOR_HIGHLIGHT_1),
            run_time=0.6, rate_func=there_and_back
        )
        
        # Highlight CV part
        self.lecture[1].set_color(COLOR_HIGHLIGHT_2)
        self.play(FadeIn(eye_icon), FadeIn(vision_grid))
        self.play(
            vision_grid[5].animate.set_fill(COLOR_HIGHLIGHT_2, opacity=0.8),
            vision_grid[10].animate.set_fill(COLOR_HIGHLIGHT_2, opacity=0.8),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A sharp image blurs by averaging pixels with neighbors.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Transition: Clear the screen for the blur demonstration
        self.play(FadeOut(pulse, bell, img_icon, audio_icon, eye_icon, vision_grid))
        
        # 8x8 Pixel Grid representing a sharp image
        sharp_pixels = VGroup(*[
            Square(side_length=0.4, stroke_width=1, color=WHITE) 
            for _ in range(64)
        ]).arrange_in_grid(8, 8, buff=0)
        
        # Create a sharp pattern (e.g., a simple cross)
        for i, p in enumerate(sharp_pixels):
            row, col = i // 8, i % 8
            if row == 4 or col == 4:
                p.set_fill(WHITE, opacity=0.9)
            else:
                p.set_fill(WHITE, opacity=0.1)
            
        # Issue 23: Change area and scale
        self.place_in_area(sharp_pixels, 'A2', 'F5', scale_factor=1.1)
        self.play(FadeIn(sharp_pixels))
        self.wait(0.5)
        
        # Visual blur effect: smooth out the colors
        blur_animations = []
        for i, p in enumerate(sharp_pixels):
            row, col = i // 8, i % 8
            if abs(row - 4) <= 1 or abs(col - 4) <= 1:
                blur_animations.append(p.animate.set_fill(GRAY_B, opacity=0.5))
            else:
                blur_animations.append(p.animate.set_fill(GRAY_C, opacity=0.2))
                
        self.play(*blur_animations, run_time=1.5)
        
        # Highlight a 3x3 region to show the averaging process
        # Focusing on index 27 (row 3, col 3)
        region_indices = [18, 19, 20, 26, 27, 28, 34, 35, 36]
        highlight_box = SurroundingRectangle(VGroup(*[sharp_pixels[i] for i in region_indices]), color=YELLOW, buff=0.05)
        
        self.play(Create(highlight_box))
        
        # Flowing "information" (dots) from neighbors to center
        center_pos = sharp_pixels[27].get_center()
        flow_dots = VGroup(*[
            Dot(sharp_pixels[i].get_center(), radius=0.05, color=YELLOW)
            for i in region_indices if i != 27
        ])
        
        self.play(
            LaggedStart(*[
                dot.animate.move_to(center_pos).set_opacity(0)
                for dot in flow_dots
            ], lag_ratio=0.1),
            Indicate(sharp_pixels[27], color=YELLOW),
            run_time=2
        )
        
        self.wait(2)
