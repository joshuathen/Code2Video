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
        title = "Summary: From Attention to Intelligence"
        lecture_lines = [
            "Attention enables models to process entire sequences at once.",
            "This parallel processing is much faster than reading word-by-word.",
            "It is the core engine driving modern AI intelligence."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors
        GRAY = "#808080"
        AQUA = "#7FFFD4"

        # === Animation for Lecture Line 1 ===
        # Lecture line index 0: "Attention enables models to process entire sequences at once."
        self.play(self.lecture[0].animate.set_color(AQUA))
        
        # Robot Icon from Asset (Issue 23 & 35)
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.set_color(GRAY)
        self.place_at_grid(robot, "B2", scale_factor=0.8)
        
        # Sentence word-by-word (Issue 36)
        words_list = ["The", "quick", "brown", "fox"]
        # Use simple Text mobjects created once
        word_mobjects = VGroup(*[Text(w, font_size=18, color=GRAY) for w in words_list]).arrange(RIGHT, buff=0.2)
        self.place_in_area(word_mobjects, "B3", "B6", scale_factor=1.0)
        
        # Magnifying glass
        mg_lens = Circle(radius=0.12, color=WHITE, stroke_width=2)
        mg_handle = Line(start=ORIGIN, end=[0.12, -0.12, 0], color=WHITE, stroke_width=2).next_to(mg_lens, DR, buff=-0.03)
        magnifying_glass = VGroup(mg_lens, mg_handle)
        magnifying_glass.move_to(word_mobjects[0].get_center())
        
        self.play(FadeIn(robot), FadeIn(word_mobjects), FadeIn(magnifying_glass))
        
        # Animate magnifying glass moving word-by-word
        for i in range(1, len(word_mobjects)):
            self.play(magnifying_glass.animate.move_to(word_mobjects[i].get_center()), run_time=0.4)
        
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture line index 1: "This parallel processing is much faster than reading word-by-word."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(AQUA)
        )
        
        # Transformer Grid Construction (Issue 37)
        # Create a 3x4 grid of dots
        grid_points = VGroup()
        for i in range(3): # Rows
            for j in range(4): # Cols
                dot = Circle(radius=0.08, color=AQUA, fill_opacity=0.8, stroke_width=1)
                dot.move_to(np.array([j * 0.8, -i * 0.8, 0])) # Local relative spacing
                grid_points.add(dot)
        
        # Connections (Web of interconnected lines)
        connections = VGroup()
        for i in range(len(grid_points)):
            for j in range(i + 1, len(grid_points)):
                line = Line(
                    grid_points[i].get_center(), 
                    grid_points[j].get_center(), 
                    color=AQUA, 
                    stroke_width=0.5, 
                    stroke_opacity=0.2
                )
                connections.add(line)
        
        transformer_visual = VGroup(connections, grid_points)
        # Position the entire visual as per Issue 37
        self.place_in_area(transformer_visual, "C2", "F5", scale_factor=1.1)
        
        # Clear previous elements and show Transformer simultaneously
        self.play(
            FadeOut(robot), FadeOut(word_mobjects), FadeOut(magnifying_glass),
            Create(transformer_visual),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture line index 2: "It is the core engine driving modern AI intelligence."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(AQUA)
        )
        
        # Pulse the entire grid with AQUA light
        # Using a flash/pulse effect on dots
        pulse_dots = grid_points.copy().set_fill(AQUA, opacity=1).set_stroke(AQUA, width=4)
        
        self.add(pulse_dots)
        self.play(
            pulse_dots.animate.scale(1.2).set_opacity(0),
            connections.animate.set_stroke(opacity=0.6, width=1),
            rate_func=there_and_back,
            run_time=2
        )
        self.remove(pulse_dots)
        
        # Final subtle glow of connections
        self.play(
            grid_points.animate.set_color(AQUA),
            connections.animate.set_stroke(opacity=0.3, width=0.5),
            run_time=1
        )
        
        self.wait(3)
