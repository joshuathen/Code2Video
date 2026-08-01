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
        self.setup_layout(
            "The Checklist: Summary & Review", 
            [
                "Review our checklist: Visuals, Logic, and Simplicity.", 
                "Test your explanations against these three criteria.", 
                "Clear communication makes math accessible for everyone."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        item1 = Text("1. Visual-Symbolic Mapping", font_size=24, color=WHITE)
        item2 = Text("2. Logical Cohesion", font_size=24, color=WHITE)
        item3 = Text("3. Parsimony", font_size=24, color=WHITE)
        
        self.place_in_area(item1, "A1", "A4")
        self.place_in_area(item2, "B1", "B4")
        self.place_in_area(item3, "C1", "C4")
        
        self.play(
            Write(item1),
            Write(item2),
            Write(item3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        def get_checkmark():
            return VGroup(
                Line(LEFT * 0.15 + DOWN * 0.1, ORIGIN, stroke_width=6),
                Line(ORIGIN, RIGHT * 0.25 + UP * 0.35, stroke_width=6)
            ).set_color("#00FF00")
        
        check1 = get_checkmark()
        check2 = get_checkmark()
        check3 = get_checkmark()
        
        self.place_at_grid(check1, "A5")
        self.place_at_grid(check2, "B5")
        self.place_at_grid(check3, "C5")
        
        self.play(Create(check1))
        self.play(Create(check2))
        self.play(Create(check3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Load Pip Asset
        try:
            pip = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/pip.svg").set_color(WHITE)
        except:
            # Fallback if asset is missing
            pip = Circle(radius=0.3, color=WHITE).add(Dot(LEFT*0.1+UP*0.1), Dot(RIGHT*0.1+UP*0.1), Arc(radius=0.1, start_angle=PI, angle=PI))
            
        self.place_at_grid(pip, "F6", scale_factor=1.2)
        
        # Clear Diagram
        diagram = VGroup(
            RoundedRectangle(corner_radius=0.1, height=1.8, width=2.5, color=WHITE),
            Line(LEFT*0.8, RIGHT*0.8, color=BLUE).shift(UP*0.3),
            Text("f(x) = y", font_size=24).shift(DOWN*0.3)
        )
        self.place_in_area(diagram, "D2", "F4", scale_factor=0.9)
        
        # Approval Stamp (Large Green Check)
        stamp = get_checkmark().scale(3).set_stroke(width=10)
        self.place_in_area(stamp, "D2", "F4")
        
        self.play(FadeIn(pip), FadeIn(diagram))
        self.play(Create(stamp))
        self.play(stamp.animate.set_opacity(0.8)) # Visual "stamp" effect
        
        self.wait(2)
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
