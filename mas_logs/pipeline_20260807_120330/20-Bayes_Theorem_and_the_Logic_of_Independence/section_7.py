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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "Summary & Key Takeaways"
        lecture_lines = [
            "- Bayes' Theorem is a tool for updating beliefs.",
            "- Always consider the prior base rate first.",
            "- Evidence is only useful if it's dependent."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors (consistent with previous sections)
        PRIOR_COLOR = "#FFFF00"
        POSTERIOR_COLOR = "#FF00FF"
        EVIDENCE_COLOR = "#00FFFF"
        HIGHLIGHT_COLOR = "#00FF00" # Likelihood/Success

        # === Animation for Lecture Line 1 ===
        # "Bayes' Theorem is a tool for updating beliefs."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        prior_text = Text("Prior", color=PRIOR_COLOR)
        posterior_text = Text("Posterior", color=POSTERIOR_COLOR)
        
        # FIX (Issue 46): Shift 'Prior' and 'Posterior' to top row (Row A)
        self.place_at_grid(prior_text, 'A2', scale_factor=0.8)
        self.place_at_grid(posterior_text, 'A5', scale_factor=0.8)
        
        arrow = Arrow(
            start=prior_text.get_right(), 
            end=posterior_text.get_left(), 
            color=WHITE, 
            buff=0.1
        )
        
        self.play(
            FadeIn(prior_text),
            FadeIn(posterior_text),
            Create(arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Always consider the prior base rate first."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        prior_highlight = SurroundingRectangle(prior_text, color=YELLOW, buff=0.1)
        self.play(Create(prior_highlight))
        self.play(Indicate(prior_text, color=YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Evidence is only useful if it's dependent."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Centered evidence text
        evidence_text = Text("Evidence", color=EVIDENCE_COLOR)
        self.place_in_area(evidence_text, "C3", "C4", scale_factor=0.8) # Moved up slightly to not clash with lower icons
        
        cross_mark = Cross(evidence_text, stroke_color=RED)
        
        self.play(FadeIn(evidence_text))
        self.play(Create(cross_mark))
        self.wait(1)

        # Final Animation: Sherlock Bones successfully identifies the 'Cat'
        # Clear specific visuals to make room for final demonstration
        self.play(
            FadeOut(prior_text), FadeOut(posterior_text), FadeOut(arrow),
            FadeOut(prior_highlight), FadeOut(evidence_text), FadeOut(cross_mark)
        )
        
        # Represent Sherlock Bones (Dog) and the Cat
        # FIX (Issue 46): Relocate icons to bottom row (Row F) and reduce scale
        dog_body = Circle(radius=0.3, color=BLUE, fill_opacity=0.3)
        dog_ears = VGroup(
            Triangle(color=BLUE, fill_opacity=0.3).scale(0.1).move_to(dog_body.get_top() + LEFT*0.1 + UP*0.05),
            Triangle(color=BLUE, fill_opacity=0.3).scale(0.1).move_to(dog_body.get_top() + RIGHT*0.1 + UP*0.05)
        )
        dog_icon = VGroup(
            VGroup(dog_body, dog_ears),
            Text("Sherlock", font_size=16)
        ).arrange(DOWN, buff=0.1)
        
        cat_icon = VGroup(
            Square(side_length=0.5, color=ORANGE, fill_opacity=0.3),
            Text("Cat", font_size=16)
        ).arrange(DOWN, buff=0.1)
        
        self.place_at_grid(dog_icon, 'F2', scale_factor=0.8)
        self.place_at_grid(cat_icon, 'F5', scale_factor=0.8)
        
        # Sherlock Bones finding the cat
        self.play(FadeIn(dog_icon), FadeIn(cat_icon))
        self.wait(0.5)
        
        # Twitching ears (Evidence)
        self.play(
            dog_ears.animate.scale(1.5, about_edge=DOWN),
            rate_func=there_and_back,
            run_time=0.5
        )
        self.play(
            dog_ears.animate.scale(1.5, about_edge=DOWN),
            rate_func=there_and_back,
            run_time=0.5
        )
        
        # Move dog to cat (finding it)
        # Using a clear path (B009) - Moving along Row F
        self.play(
            dog_icon.animate.move_to(self.grid["F4"]),
            run_time=0.8
        )
        self.play(
            dog_icon.animate.move_to(self.grid["F5"] + LEFT*0.7),
            run_time=0.4
        )
        
        # Success celebration
        success_star = Star(n=5, color=GOLD, fill_opacity=0.8).scale(0.4)
        # Position star above the discovery (at Row E)
        success_star.move_to(self.grid["E5"])
        
        self.play(
            Create(success_star),
            dog_body.animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(
            success_star.animate.scale(1.5).rotate(PI/4),
            rate_func=there_and_back
        )
        
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
