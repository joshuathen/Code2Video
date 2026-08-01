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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines for Section 2
        title_text = "Prerequisite: Dot Products as Similarity"
        lecture_lines = [
            "How does the model \"search\" for a specific fact?",
            "Dot products measure similarity between two vectors.",
            "High similarity triggers the retrieval of stored information."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define grid points for vector origin and axes
        origin_pt = self.grid["D2"]
        a_end_pt = self.grid["B2"]
        b_end_pt = self.grid["D4"]
        
        # === Animation for Lecture Line 1 ===
        # Display two vectors 'a' and 'b' at 90-degree angle (#FFFFFF) 
        # accompanied by the [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/search.svg] icon.
        self.lecture[0].set_color(WHITE)
        
        vec_a = Arrow(origin_pt, a_end_pt, color=WHITE, buff=0)
        vec_b = Arrow(origin_pt, b_end_pt, color=WHITE, buff=0)
        
        label_a = MathTex("a", color=WHITE)
        self.place_at_grid(label_a, "B1")
        
        label_b = MathTex("b", color=WHITE)
        # Resolved Issue 30: label_b at D4
        self.place_at_grid(label_b, "D4")
        label_b.shift(RIGHT * 0.4)
        
        # Load search icon asset
        # Resolved Issue 29: search_icon at B2
        search_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/search.svg")
        self.place_at_grid(search_icon, "B2", scale_factor=0.6)
        search_icon.shift(UP * 0.5)
        
        self.play(
            Create(vec_a),
            Create(vec_b),
            Write(label_a),
            Write(label_b),
            FadeIn(search_icon),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate vector 'b' to align with 'a' and turn green (#00FF00).
        self.lecture[1].set_color(WHITE)
        
        target_b_label_pos = self.grid["B3"]
        
        self.play(
            Rotate(vec_b, angle=PI/2, about_point=origin_pt),
            vec_b.animate.set_color("#00FF00"),
            label_b.animate.move_to(target_b_label_pos).set_color("#00FF00"),
            self.lecture[1].animate.set_color("#00FF00"),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a 'Match Score' bar filling up as alignment increases (#00FF00)
        # alongside the [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fact.svg] icon.
        self.lecture[2].set_color("#00FF00")
        
        # Bar container at bottom area
        bar_bg = Rectangle(width=3.0, height=0.3, color=WHITE).set_stroke(width=2)
        self.place_in_area(bar_bg, "F2", "F5")
        
        # Fill mobject
        bar_fill = Rectangle(width=0.01, height=0.25, color="#00FF00", fill_opacity=0.8).set_stroke(width=0)
        bar_fill.move_to(bar_bg.get_left(), aligned_edge=LEFT)
        
        score_label = Text("Match Score", font_size=20, color=WHITE)
        self.place_at_grid(score_label, "F1")
        score_label.shift(RIGHT * 0.5)
        
        # Load fact icon asset
        # Resolved Issue 31: fact_icon at D5
        fact_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fact.svg")
        self.place_at_grid(fact_icon, "D5", scale_factor=0.6)
        
        self.play(
            Create(bar_bg),
            Write(score_label),
            FadeIn(fact_icon),
            run_time=1
        )
        
        fill_tracker = ValueTracker(0.01)
        bar_fill.add_updater(lambda m: m.stretch_to_fit_width(fill_tracker.get_value(), about_edge=LEFT))
        self.add(bar_fill)
        
        self.play(
            fill_tracker.animate.set_value(3.0),
            run_time=2,
            rate_func=smooth
        )
        bar_fill.clear_updaters()
        
        # Retrieval trigger visual
        self.play(
            Indicate(fact_icon, color="#00FF00", scale_factor=1.2),
            run_time=1
        )
        
        self.wait(2)
