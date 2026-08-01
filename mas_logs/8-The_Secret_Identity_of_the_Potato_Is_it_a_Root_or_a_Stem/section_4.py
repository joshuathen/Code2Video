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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The 'Eyes' Have It: Examining the Potato", [
            "Look closely at the \"eyes\" on a potato's surface.",
            "These eyes are actually nodes with axillary buds.",
            "In the dark, these buds sprout into new stems.",
            "True roots, like carrots, cannot sprout this way.",
            "This structural evidence proves the potato is a stem."
        ])

        # === Animation for Lecture Line 1 ===
        # Visual: A large tan potato (#D2B48C) appears with dark brown spots (#4B3621) labeled 'Eyes'.
        self.lecture[0].set_color(YELLOW)
        
        potato_body = Ellipse(width=3, height=2, color="#D2B48C", fill_opacity=1)
        # Position spots relative to the potato center before placing the group
        eye1 = Dot(point=LEFT*0.7 + UP*0.3, color="#4B3621", radius=0.08)
        eye2 = Dot(point=RIGHT*0.8 + DOWN*0.2, color="#4B3621", radius=0.08)
        eye3 = Dot(point=LEFT*0.2 + DOWN*0.5, color="#4B3621", radius=0.08)
        potato_group = VGroup(potato_body, eye1, eye2, eye3)
        
        self.place_in_area(potato_group, "A2", "D4", scale_factor=1.0)
        
        eyes_label = Text("Eyes", font_size=24, color="#4B3621")
        self.place_at_grid(eyes_label, "A2") # Near the top of the potato
        
        self.play(FadeIn(potato_group), Write(eyes_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: Blue circles (#00BFFF) highlight the 'eyes,' and the label updates to 'Nodes with Buds'.
        self.lecture[1].set_color(YELLOW)
        
        highlight1 = Circle(radius=0.15, color="#00BFFF").move_to(eye1.get_center())
        highlight2 = Circle(radius=0.15, color="#00BFFF").move_to(eye2.get_center())
        highlight3 = Circle(radius=0.15, color="#00BFFF").move_to(eye3.get_center())
        highlights = VGroup(highlight1, highlight2, highlight3)
        
        nodes_label = Text("Nodes with Buds", font_size=24, color="#00BFFF")
        self.place_at_grid(nodes_label, "A2")
        
        self.play(
            Create(highlights),
            Transform(eyes_label, nodes_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: Bright green sprouts (#32CD32) grow outwards from several of the highlighted 'eyes'.
        self.lecture[2].set_color(YELLOW)
        
        sprout1 = Line(eye1.get_center(), eye1.get_center() + LEFT*0.6 + UP*0.4, color="#32CD32", stroke_width=6)
        sprout2 = Line(eye2.get_center(), eye2.get_center() + RIGHT*0.7 + DOWN*0.3, color="#32CD32", stroke_width=6)
        sprout3 = Line(eye3.get_center(), eye3.get_center() + DOWN*0.5 + LEFT*0.2, color="#32CD32", stroke_width=6)
        sprouts = VGroup(sprout1, sprout2, sprout3)
        
        self.play(Create(sprouts))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Visual: An orange carrot (#FFA500) appears next to the potato; it remains static.
        self.lecture[3].set_color(YELLOW)
        
        # Reposition potato to make room for carrot
        self.play(potato_group.animate.scale(0.7), sprouts.animate.scale(0.7), highlights.animate.scale(0.7), eyes_label.animate.scale(0.7))
        # Recalculate positions manually since we can't use move_to directly comfortably after initial placement
        # Shift entire assembly to the left grid area
        target_pos_potato = self.grid["B2"]
        self.play(
            potato_group.animate.move_to(target_pos_potato),
            sprouts.animate.move_to(target_pos_potato),
            highlights.animate.move_to(target_pos_potato),
            eyes_label.animate.move_to(self.grid["A2"])
        )

        carrot_body = Polygon(
            [-1, 1.5, 0], [1, 1.5, 0], [0, -1.5, 0],
            color="#FFA500", fill_opacity=1
        )
        carrot_top = Arc(radius=0.3, start_angle=0, angle=PI, color="#32CD32", fill_opacity=1).move_to(carrot_body.get_top())
        carrot_group = VGroup(carrot_body, carrot_top)
        
        self.place_in_area(carrot_group, "B5", "E6", scale_factor=0.6)
        carrot_label = Text("Carrot (Root)", font_size=24, color="#FFA500")
        self.place_at_grid(carrot_label, "A5")
        
        self.play(FadeIn(carrot_group), Write(carrot_label))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Visual: The potato with sprouts is positioned next to the stem diagram to show matching node patterns.
        self.lecture[4].set_color(YELLOW)
        
        # Create a simplified Stem Diagram (from context of Section 2)
        stem_line = Line(DOWN*1.5, UP*1.5, color=WHITE, stroke_width=4)
        n1 = Dot(stem_line.point_from_proportion(0.2), color="#00BFFF")
        n2 = Dot(stem_line.point_from_proportion(0.5), color="#00BFFF")
        n3 = Dot(stem_line.point_from_proportion(0.8), color="#00BFFF")
        stem_diag = VGroup(stem_line, n1, n2, n3)
        stem_diag_label = Text("Stem Structure", font_size=20, color=WHITE)
        
        # Placement for comparison
        self.play(FadeOut(carrot_group), FadeOut(carrot_label))
        
        self.place_in_area(stem_diag, "B5", "E5", scale_factor=0.8)
        self.place_at_grid(stem_diag_label, "A5")
        
        stem_label_final = Text("Stem", font_size=32, color=YELLOW, weight=BOLD)
        self.place_at_grid(stem_label_final, "F4")
        
        self.play(
            Create(stem_diag),
            Write(stem_diag_label),
            Write(stem_label_final)
        )
        self.wait(2)
