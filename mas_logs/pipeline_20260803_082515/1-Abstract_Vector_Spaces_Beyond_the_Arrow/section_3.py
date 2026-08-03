from manim import *

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
        self.setup_layout(
            "The 8 Golden Rules (Axioms)",
            [
                "To qualify, a set must obey ten strict axioms.",
                "Closure ensures results stay within the same set.",
                "Every space needs a zero vector as an origin.",
                "Every element must have an additive inverse.",
                "These rules create a consistent mathematical playground."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Display a checklist [Asset: checklist.svg] with '8 Axioms' title.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        checklist_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/checklist.svg")
        self.place_at_grid(checklist_icon, "A2", scale_factor=0.5)
        
        checklist_title = Text("8 Axioms Checklist", font_size=24, color=WHITE)
        self.place_at_grid(checklist_title, "A4", scale_factor=0.8)
        
        items = ["1. Closure", "2. Associativity", "3. Commutativity", "4. Zero Vector", "5. Inverses"]
        checklist_items = VGroup(*[Text(item, font_size=20, color=WHITE) for item in items]).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(checklist_items, "B3", "D5", scale_factor=0.8)
        
        self.play(FadeIn(checklist_icon), Create(checklist_title), Create(checklist_items))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # First item 'Closure' highlights in gold (#FFD700).
        self.play(
            self.lecture[1].animate.set_color("#FFD700"),
            checklist_items[0].animate.set_color("#FFD700")
        )
        
        boundary = Circle(radius=1.5, color="#0000FF")
        self.place_in_area(boundary, "E2", "F4", scale_factor=0.7)
        
        # Dots relative to boundary center
        center_b = boundary.get_center()
        dot1 = Dot(center_b + LEFT*0.4 + UP*0.3, color=WHITE)
        dot2 = Dot(center_b + RIGHT*0.5 + DOWN*0.2, color=WHITE)
        plus = MathTex("+", font_size=30, color=WHITE).move_to((dot1.get_center() + dot2.get_center())/2 + UP*0.3)
        
        result_dot = Dot(center_b + DOWN*0.4, color="#FFD700")
        
        self.play(Create(boundary))
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.play(Write(plus))
        self.play(Transform(VGroup(dot1, dot2, plus), result_dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every space needs a zero vector as an origin.
        self.play(
            self.lecture[2].animate.set_color("#FF0000"),
            checklist_items[3].animate.set_color("#FF0000")
        )
        
        zero_vec = MathTex(r"\vec{0}", font_size=40, color="#FF0000")
        zero_vec.move_to(center_b) 
        
        self.play(Write(zero_vec))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Every element must have an additive inverse.
        self.play(
            self.lecture[3].animate.set_color("#FF0000"),
            checklist_items[4].animate.set_color("#FF0000")
        )
        
        # Fixed: Using fill_opacity=0 instead of opacity=0 for Dot constructor
        inv_area_center = self.place_in_area(Dot(fill_opacity=0), "E5", "F6").get_center()
        
        v_vec = Arrow(start=inv_area_center, end=inv_area_center + RIGHT*0.6 + UP*0.5, color="#FF0000", buff=0)
        mv_vec = Arrow(start=inv_area_center, end=inv_area_center + LEFT*0.6 + DOWN*0.5, color="#FF0000", buff=0)
        v_label = MathTex("v", color="#FF0000", font_size=20).next_to(v_vec.get_end(), UR, buff=0.1)
        mv_label = MathTex("-v", color="#FF0000", font_size=20).next_to(mv_vec.get_end(), DL, buff=0.1)
        
        v_group = VGroup(v_vec, v_label)
        mv_group = VGroup(mv_vec, mv_label)
        
        v_group.scale(0.6)
        mv_group.scale(0.6)
        
        self.play(Create(v_group), Create(mv_group))
        self.wait(0.5)
        
        # Move them together and vanish into '0' symbol
        self.play(
            v_group.animate.move_to(zero_vec.get_center()).set_opacity(0),
            mv_group.animate.move_to(zero_vec.get_center()).set_opacity(0),
            zero_vec.animate.scale(1.5),
            run_time=1.5
        )
        self.play(zero_vec.animate.scale(1/1.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # These rules create a consistent mathematical playground.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Flash all checklist items as valid (green)
        self.play(*[item.animate.set_color(GREEN) for item in checklist_items])
        self.wait(2)
