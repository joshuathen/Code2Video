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
        # Colors
        V_COLOR = "#00FF00"  # Green
        V2_COLOR = "#FFFF00" # Yellow
        W_COLOR = "#0000FF"  # Blue
        SUM_COLOR = "#FF00FF" # Magenta

        self.setup_layout(
            "Prerequisite: Vectors as Movements", 
            [
                'Vectors are simple instructions for movement in space.', 
                'Scaling a vector stretches or shrinks its path.', 
                'Adding vectors combines multiple movements into one.'
            ]
        )
        
        # Grid/Coordinate system background
        plane = NumberPlane(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'A1', 'F6')
        self.add(plane)

        # Vector origin point in the grid space
        origin = plane.c2p(1, 1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(V_COLOR))
        
        # Vector v: (1,1) to (3,2) [Delta = (2,1)]
        v_arrow = Arrow(
            start=origin,
            end=plane.c2p(3, 2),
            buff=0,
            color=V_COLOR,
            stroke_width=6
        )
        label_v = Text("v", font_size=24, color=V_COLOR)
        # Position label v near its tip
        self.place_at_grid(label_v, 'D4', scale_factor=0.6)
        
        self.play(GrowArrow(v_arrow), FadeIn(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(V2_COLOR)
        )
        
        # Vector 2v: (1,1) to (5,3) [Delta = (4,2)]
        v2_arrow = Arrow(
            start=origin,
            end=plane.c2p(5, 3),
            buff=0,
            color=V2_COLOR,
            stroke_width=6
        )
        label_2v = Text("2v", font_size=24, color=V2_COLOR)
        # Issue 29: Position 2v label at C5
        self.place_at_grid(label_2v, 'C5', scale_factor=0.6)
        
        self.play(
            Transform(v_arrow, v2_arrow),
            FadeOut(label_v),
            FadeIn(label_2v)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(W_COLOR)
        )
        
        # Vector w: origin(1,1) to (2, 0.5) instruction is (1, -0.5)
        w_start_pos = origin
        w_end_pos = plane.c2p(2, 0.5)
        w_arrow = Arrow(
            start=w_start_pos,
            end=w_end_pos,
            buff=0,
            color=W_COLOR,
            stroke_width=6
        )
        label_w = Text("w", font_size=24, color=W_COLOR)
        # Issue 28: Position w label at D6
        self.place_at_grid(label_w, 'D6', scale_factor=0.6)
        
        self.play(GrowArrow(w_arrow))
        self.wait(0.5)
        
        # Move w to the tip of 2v (5, 3)
        # new tip is (5+1, 3-0.5) = (6, 2.5)
        new_w_start = plane.c2p(5, 3)
        new_w_end = plane.c2p(6, 2.5)
        
        moved_w_arrow = Arrow(
            start=new_w_start,
            end=new_w_end,
            buff=0,
            color=W_COLOR,
            stroke_width=6
        )

        self.play(
            ReplacementTransform(w_arrow, moved_w_arrow),
            FadeIn(label_w)
        )
        
        # Final Sum Vector from origin (1,1) to new_w_end (6, 2.5)
        sum_arrow = Arrow(
            start=origin,
            end=new_w_end,
            buff=0,
            color=SUM_COLOR,
            stroke_width=8
        )
        label_sum = Text("2v + w", font_size=24, color=SUM_COLOR)
        # Issue 27: Position sum label in area E4-F5
        self.place_in_area(label_sum, 'E4', 'F5', scale_factor=0.5)
        
        self.play(GrowArrow(sum_arrow), FadeIn(label_sum))
        self.wait(2)
