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
        # Setup layout with specific lines
        lecture_lines = [
            "Convolution merges two sequences into a new signal.",
            "Imagine a clear Pixel Fox entering a foggy forest.",
            "The fog's density modifies the fox's sharp image.",
            "This interaction blends the data into a blurred output.",
            "Mathematically, one signal influences another over time."
        ]
        self.setup_layout("Introduction: The Concept of Merging Information", lecture_lines)

        # Colors
        BLUE_BAR = "#58C4DD"
        GREEN_BAR = "#83C167"
        PURPLE_BAR = "#966FD6"
        FOG_GRAY = "#A9A9A9"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_BAR))
        
        # Create sequences
        blue_bars = VGroup(*[
            Rectangle(width=0.2, height=h, fill_opacity=0.8, fill_color=BLUE_BAR, stroke_width=1) 
            for h in [0.5, 1.0, 1.5, 1.0, 0.5]
        ]).arrange(RIGHT, buff=0.1)
        
        green_bars = VGroup(*[
            Rectangle(width=0.2, height=h, fill_opacity=0.8, fill_color=GREEN_BAR, stroke_width=1) 
            for h in [0.8, 1.2, 0.8]
        ]).arrange(RIGHT, buff=0.1)

        purple_bars = VGroup(*[
            Rectangle(width=0.2, height=h, fill_opacity=0.8, fill_color=PURPLE_BAR, stroke_width=1) 
            for h in [0.4, 0.9, 1.5, 2.0, 1.5, 0.9, 0.4]
        ]).arrange(RIGHT, buff=0.1)

        self.place_at_grid(blue_bars, "A2")
        self.place_at_grid(green_bars, "A5")
        
        self.play(Create(blue_bars), Create(green_bars))
        self.wait(0.5)
        
        # Move towards each other and transform
        self.play(
            blue_bars.animate.move_to(self.grid["A3"] + RIGHT*0.5),
            green_bars.animate.move_to(self.grid["A3"] + RIGHT*0.5),
            run_time=1.5
        )
        self.place_at_grid(purple_bars, "A3", scale_factor=1.0)
        purple_bars.shift(RIGHT*0.5) # Align with target
        self.play(ReplacementTransform(VGroup(blue_bars, green_bars), purple_bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Load Pixel Fox
        fox_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/fox.svg"
        try:
            fox = SVGMobject(fox_path)
        except:
            # Fallback if asset missing
            fox = Square(color=ORANGE, fill_opacity=1).scale(0.5)
        
        self.place_in_area(fox, "B1", "D3", scale_factor=1.5)
        self.play(DrawBorderThenFill(fox))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(FOG_GRAY))
        
        # Fog pattern (semi-transparent gray rectangles)
        fog_rects = VGroup(*[
            Rectangle(width=0.4, height=0.4, fill_color=FOG_GRAY, fill_opacity=0.4, stroke_width=0)
            for _ in range(12)
        ])
        for i, rect in enumerate(fog_rects):
            row_idx = i // 4
            col_idx = i % 4
            target_pos = self.grid["B1"] + np.array([col_idx * 0.6, -row_idx * 0.6, 0])
            rect.move_to(target_pos)

        self.play(FadeIn(fog_rects, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE_BAR))
        
        # Blur interaction - reduce fox opacity and add more fog
        self.play(
            fox.animate.set_opacity(0.4),
            fog_rects.animate.set_opacity(0.6).scale(1.2),
            run_time=1.5
        )
        self.wait(1)

        # Clean up fox/fog for final math concept
        self.play(FadeOut(fox), FadeOut(fog_rects), FadeOut(purple_bars))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN_BAR))
        
        # Sliding window demonstration
        # Base signal
        signal_data = [0.5, 1.2, 0.8, 1.5, 1.0, 0.7, 1.3]
        base_signal = VGroup(*[
            Rectangle(width=0.3, height=h, fill_opacity=0.5, fill_color=WHITE, stroke_width=1)
            for h in signal_data
        ]).arrange(RIGHT, buff=0.2)
        self.place_at_grid(base_signal, "E3", scale_factor=1.2)
        
        # Moving window
        window = Rectangle(width=1.0, height=2.0, stroke_color=YELLOW, stroke_width=3, fill_opacity=0.2, fill_color=YELLOW)
        window.move_to(base_signal[0].get_center())
        
        # Points to be plotted
        points = VGroup()
        
        self.play(Create(base_signal))
        self.play(Create(window))
        
        # Animation loop for sliding window
        for i in range(len(signal_data)):
            # Calculate "average" for visual plotting
            avg_height = signal_data[i]
            point = Dot(color=GREEN_BAR).move_to(base_signal[i].get_top() + UP*0.3)
            points.add(point)
            
            if i < len(signal_data) - 1:
                self.play(
                    window.animate.move_to(base_signal[i+1].get_center()),
                    Create(point),
                    run_time=0.6
                )
            else:
                self.play(Create(point), run_time=0.6)
        
        self.play(Create(Line(points[0].get_center(), points[-1].get_center(), color=GREEN_BAR)))
        self.wait(2)
