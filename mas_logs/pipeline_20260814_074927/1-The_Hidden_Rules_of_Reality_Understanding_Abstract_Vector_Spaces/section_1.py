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
        self.setup_layout("Prerequisite: The World of Arrows", [
            "Vectors are often seen as arrows on a grid.",
            "We can add arrows using the tip-to-tail method.",
            "Scaling an arrow changes its length but not direction."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Prominent color: Green (#00FF00) for vector 'v'
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # 2D Grid background
        plane = NumberPlane(
            x_range=[-1, 7, 1],
            y_range=[-1, 5, 1],
            background_line_style={"stroke_color": "#FFFFFF", "stroke_opacity": 0.4},
            axis_config={"stroke_color": "#FFFFFF"}
        )
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.6)
        
        # Vector-Bot (Origin) - Using a simple icon-like circle
        vector_bot = VGroup(
            Circle(radius=0.15, color=WHITE, fill_opacity=1),
            Dot(radius=0.03, color=BLACK).move_to(plane.c2p(0.05, 0.05) - plane.c2p(0,0)),
            Dot(radius=0.03, color=BLACK).move_to(plane.c2p(-0.05, 0.05) - plane.c2p(0,0))
        )
        vector_bot.move_to(plane.c2p(0, 0))
        
        self.play(FadeIn(plane), FadeIn(vector_bot))
        
        # Arrow 'v' (0,0) to (3,2)
        v_arrow = Arrow(plane.c2p(0, 0), plane.c2p(3, 2), buff=0, color="#00FF00")
        v_label = MathTex(r"v = (3, 2)", font_size=24, color="#00FF00")
        v_label.next_to(v_arrow.get_end(), UR, buff=0.1)
        
        self.play(GrowArrow(v_arrow), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Prominent color: Yellow (#FFD700) for resultant
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Arrow 'w' from tip of 'v' (3,2) to (3,2)+(1,2) = (4,4)
        w_arrow = Arrow(plane.c2p(3, 2), plane.c2p(4, 4), buff=0, color="#00FFFF")
        w_label = MathTex(r"w", font_size=24, color="#00FFFF")
        w_label.next_to(w_arrow.get_end(), RIGHT, buff=0.1)
        
        # Resultant arrow 'v+w' from origin to (4,4)
        res_arrow = Arrow(plane.c2p(0, 0), plane.c2p(4, 4), buff=0, color="#FFD700")
        res_label = MathTex(r"v+w = (4, 4)", font_size=24, color="#FFD700")
        res_label.next_to(res_arrow.get_center(), UL, buff=0.1)
        
        self.play(GrowArrow(w_arrow), Write(w_label))
        self.play(GrowArrow(res_arrow), Write(res_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Prominent color: OrangeRed (#FF4500) for 2v
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF4500")
        )
        
        # Fade out addition elements to focus on scaling v
        self.play(FadeOut(w_arrow), FadeOut(w_label), FadeOut(res_arrow), FadeOut(res_label))
        
        # Scale v to 2v (0,0) to (6,4)
        v2_arrow = Arrow(plane.c2p(0, 0), plane.c2p(6, 4), buff=0, color="#FF4500")
        v2_label = MathTex(r"2v = (6, 4)", font_size=24, color="#FF4500")
        v2_label.next_to(v2_arrow.get_end(), UR, buff=0.1)
        
        self.play(
            Transform(v_arrow, v2_arrow),
            Transform(v_label, v2_label)
        )
        self.wait(1)
        
        # Final part of line 3: Fade out and show "Vector behavior"
        concept_text = Text("Vector behavior", font_size=36, color=WHITE)
        self.place_in_area(concept_text, "C2", "D5")
        
        self.play(
            FadeOut(plane),
            FadeOut(vector_bot),
            FadeOut(v_arrow),
            FadeOut(v_label),
            Write(concept_text)
        )
        self.wait(2)
