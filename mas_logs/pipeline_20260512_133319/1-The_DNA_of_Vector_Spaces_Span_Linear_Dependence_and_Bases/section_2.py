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
        # Setup layout with teaching content
        lecture_lines = [
            'Start with two distinct movement vectors, v and w.',
            'Scaling both vectors creates new instructions for travel.',
            'Their sum forms a unique linear combination.'
        ]
        self.setup_layout("Linear Combinations: Mixing the Ingredients", lecture_lines)

        # Vector Definitions
        v_coords = np.array([1.5, 0.5, 0])
        w_coords = np.array([0.5, 1.5, 0])
        
        # Setup Coordinate Plane
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        # Issue 30 Fix: Adjust plane position to avoid overlap with A3
        self.place_in_area(plane, 'B1', 'F6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        v_vec = Arrow(plane.c2p(0,0,0), plane.c2p(*v_coords), buff=0, color="#00FF00")
        w_vec = Arrow(plane.c2p(0,0,0), plane.c2p(*w_coords), buff=0, color="#0000FF")
        v_label = Text("v", color="#00FF00", font_size=20).next_to(v_vec.get_end(), RIGHT, buff=0.1)
        w_label = Text("w", color="#0000FF", font_size=20).next_to(w_vec.get_end(), UP, buff=0.1)
        
        # Issue 25: Incorporate [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/travel.svg] icon
        icon_v = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/travel.svg", color=WHITE).scale(0.12)
        icon_w = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/travel.svg", color=WHITE).scale(0.12)
        icon_v.move_to(v_vec.get_center() + UP*0.2)
        icon_w.move_to(w_vec.get_center() + LEFT*0.2)

        self.play(Create(plane))
        self.play(
            GrowArrow(v_vec), GrowArrow(w_vec), 
            FadeIn(v_label), FadeIn(w_label),
            FadeIn(icon_v), FadeIn(icon_w)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Scale v by 0.5 and w by 1.5
        v_scaled_coords = v_coords * 0.5
        w_scaled_coords = w_coords * 1.5
        
        v_scaled_vec = Arrow(plane.c2p(0,0,0), plane.c2p(*v_scaled_coords), buff=0, color="#00FF00")
        w_scaled_vec = Arrow(plane.c2p(0,0,0), plane.c2p(*w_scaled_coords), buff=0, color="#0000FF")
        
        # Dashed lines for the scaled versions (representing the scalar components)
        dash_v = DashedLine(plane.c2p(0,0,0), plane.c2p(*v_scaled_coords), color="#00FF00", stroke_opacity=0.6)
        dash_w = DashedLine(plane.c2p(0,0,0), plane.c2p(*w_scaled_coords), color="#0000FF", stroke_opacity=0.6)

        v_scaled_label = Text("0.5v", color="#00FF00", font_size=18).next_to(v_scaled_vec.get_end(), DOWN, buff=0.1)
        w_scaled_label = Text("1.5w", color="#0000FF", font_size=18).next_to(w_scaled_vec.get_end(), LEFT, buff=0.1)

        self.play(
            ReplacementTransform(v_vec, v_scaled_vec),
            ReplacementTransform(w_vec, w_scaled_vec),
            ReplacementTransform(v_label, v_scaled_label),
            ReplacementTransform(w_label, w_scaled_label),
            FadeIn(dash_v), FadeIn(dash_w),
            icon_v.animate.move_to(v_scaled_vec.get_center() + UP*0.2),
            icon_w.animate.move_to(w_scaled_vec.get_center() + LEFT*0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF00FF")
        
        # Result vector calculation
        res_coords = v_scaled_coords + w_scaled_coords
        res_vec = Arrow(plane.c2p(0,0,0), plane.c2p(*res_coords), buff=0, color="#FF00FF")
        
        # Parallelogram rule lines
        side1 = DashedLine(plane.c2p(*v_scaled_coords), plane.c2p(*res_coords), color=WHITE, stroke_opacity=0.5)
        side2 = DashedLine(plane.c2p(*w_scaled_coords), plane.c2p(*res_coords), color=WHITE, stroke_opacity=0.5)
        
        # Result formula label
        # Issue 31 Fix: Place in area A2 to A5 to avoid inadequate positioning
        res_formula = Text("0.5v + 1.5w", color="#FF00FF", font_size=24)
        self.place_in_area(res_formula, 'A2', 'A5', scale_factor=0.9)
        
        self.play(Create(side1), Create(side2))
        self.play(GrowArrow(res_vec), Write(res_formula))
        self.wait(2)
