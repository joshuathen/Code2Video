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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "The Transformer Architecture: Parallel Power"
        lines = [
            "Older models worked like a slow conveyor belt.",
            "Transformers work like a high-speed panoramic camera.",
            "This parallel processing allows training on the entire internet.",
            "High-speed training enables models to become incredibly massive.",
            "Parallelism is the secret to modern AI performance."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#808080"))
        
        belt_color = "#808080"
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/con.svg
        belt = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/con.svg").set_color(belt_color)
        self.place_in_area(belt, "C1", "C6", scale_factor=1.5)
        
        # Issue 47: Position belt_label at E3
        belt_label = Text("Old Models (RNN)", font_size=18, color=belt_color)
        self.place_at_grid(belt_label, "E3", scale_factor=0.8)
        
        words = ["The", "cat", "sat", "on", "the", "mat"]
        word_mobjects = [Text(w, font_size=20, color=WHITE) for w in words]
        
        self.play(Create(belt), Write(belt_label))
        
        for i, word_mob in enumerate(word_mobjects):
            # Place at left grid of belt
            self.place_at_grid(word_mob, "C1")
            # Move through belt one by one
            self.play(word_mob.animate.move_to(self.grid["C6"]), run_time=0.6)
            if i < len(word_mobjects) - 1:
                self.remove(word_mob)
        
        # Cleanup line 1
        self.play(FadeOut(belt), FadeOut(belt_label), FadeOut(word_mobjects[-1]))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        sentence = Text("The cat sat on the mat", font_size=24, color=WHITE)
        self.place_in_area(sentence, "B1", "B6")
        
        # Issue 45: Flash at A3, scaled to 0.6
        flash = Star(n=12, outer_radius=1.2, inner_radius=0.8, color=WHITE, fill_opacity=0.8)
        self.place_at_grid(flash, "A3", scale_factor=0.6)
        
        self.add(sentence)
        self.play(FadeIn(flash))
        self.play(flash.animate.scale(1.5).set_opacity(0), run_time=0.4)
        self.remove(flash)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF9900"))
        
        layer_color = "#FF9900"
        layers = VGroup(*[
            Rectangle(width=5, height=0.3, fill_color=layer_color, fill_opacity=0.3, color=layer_color)
            for _ in range(5)
        ]).arrange(DOWN, buff=0.2)
        self.place_in_area(layers, "C1", "E6")
        
        # Parallel movement of sentence through layers
        self.play(Create(layers))
        self.play(
            sentence.animate.move_to(self.grid["C3"]),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF9900"))
        
        # Draw parallel vertical arrows
        arrows = VGroup(*[
            Arrow(start=self.grid["B" + str(i)], end=self.grid["F" + str(i)], color=layer_color, stroke_width=2)
            for i in range(1, 7)
        ])
        
        self.play(Create(arrows))
        self.play(arrows.animate.set_opacity(0.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        
        # Issue 46: summary_text scaled to 0.6 to fit area
        summary_text = Text("Parallelism enables massive scale and speed", font_size=22, color="#00FFFF")
        self.place_in_area(summary_text, "F1", "F6", scale_factor=0.6)
        
        self.play(Write(summary_text))
        self.wait(2)
