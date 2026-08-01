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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout("The Nature of 'e': Continuous Growth", [
            "The constant e represents the limit of continuous growth.",
            "Imagine a value growing at a 100% annual rate.",
            "Compounding more frequently increases the final return slightly.",
            "Compounding every instant leads us to the value e.",
            "It is the base of all natural growth processes."
        ])

        # === Animation for Lecture Line 1 ===
        # Replaced MathTex with Text to avoid LaTeX dependency error
        e_symbol = Text("e", color="#FFD700", slant=ITALIC)
        e_label = Text("Base of Growth", font_size=24, color="#FFFFFF")
        
        self.place_at_grid(e_symbol, "B2", scale_factor=2.5)
        self.place_at_grid(e_label, "A2", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color("#FFD700"),
            Write(e_symbol),
            FadeIn(e_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a vertical bar of height 1 (#FFFFFF) and grow it to height 2.
        bar = Rectangle(height=1.0, width=0.6, color=WHITE, fill_opacity=0.6)
        self.place_at_grid(bar, "D2")
        
        self.play(
            self.lecture[1].animate.set_color("#FFD700"),
            Create(bar)
        )
        
        bar_height_2 = Rectangle(height=2.0, width=0.6, color=WHITE, fill_opacity=0.6)
        self.place_at_grid(bar_height_2, "D2")
        
        self.play(Transform(bar, bar_height_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Split the bar into multiple compounding segments, increasing its total height to 2.44.
        segments = VGroup(*[
            Rectangle(height=2.44/4, width=0.6, color="#00FFFF", fill_opacity=0.7, stroke_width=1)
            for _ in range(4)
        ]).arrange(UP, buff=0.05)
        self.place_at_grid(segments, "D2")
        
        self.play(
            self.lecture[2].animate.set_color("#FFD700"),
            Transform(bar, segments)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate a smooth growth curve reaching height 2.718, labeled as 'e' (#FFD700).
        axes = Axes(
            x_range=[0, 1.2, 1],
            y_range=[0, 3, 1],
            x_length=2.5,
            y_length=3,
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_at_grid(axes, "D5")
        
        # e^x curve from x=0 to x=1
        growth_curve = axes.plot(lambda x: np.exp(x), x_range=[0, 1], color="#FFD700")
        # Replaced MathTex with Text to avoid LaTeX dependency error
        e_val_text = Text("e ≈ 2.718", color="#FFD700", font_size=32)
        self.place_at_grid(e_val_text, "B5")
        
        self.play(
            self.lecture[3].animate.set_color("#FFD700"),
            Create(axes),
            Create(growth_curve),
            Write(e_val_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pulse the value 2.718 and the 'e' symbol to emphasize it as the natural limit.
        self.play(
            self.lecture[4].animate.set_color("#FFD700"),
            e_val_text.animate.scale(1.3),
            e_symbol.animate.scale(1.3)
        )
        self.play(
            e_val_text.animate.scale(1/1.3),
            e_symbol.animate.scale(1/1.3)
        )
        self.wait(2)
