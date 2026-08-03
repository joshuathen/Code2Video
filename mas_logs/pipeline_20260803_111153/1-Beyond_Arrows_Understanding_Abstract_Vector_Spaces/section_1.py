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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the given title and lecture lines
        self.setup_layout(
            "Prerequisite: The Familiar Vector", 
            [
                "We often view vectors as arrows in 2D space.",
                "Adding vectors follows the simple tip-to-tail rule.",
                "Scaling a vector stretches or shrinks its length."
            ]
        )
        
        # Define Colors
        color_u = "#0000FF"
        color_v = "#FFFF00"
        color_uv = "#00FF00"
        color_scaled_u = "#ADD8E6"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(color_u)

        # Create coordinate system
        # Use a localized NumberPlane within the grid area
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        # Shifted from A1 to B2 to avoid crowding and clipping (Issue #25)
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.6)
        origin = plane.get_origin()

        # Define vectors relative to plane coordinates
        vec_u_coords = [1, 2, 0]
        vec_v_coords = [2, -1, 0]

        arrow_u = Arrow(
            start=origin, 
            end=plane.c2p(*vec_u_coords), 
            buff=0, 
            color=color_u, 
            stroke_width=6
        )
        label_u = MathTex(r"\vec{u}", color=color_u, font_size=24)
        label_u.next_to(arrow_u.get_end(), UR, buff=0.1)

        arrow_v = Arrow(
            start=origin, 
            end=plane.c2p(*vec_v_coords), 
            buff=0, 
            color=color_v, 
            stroke_width=6
        )
        label_v = MathTex(r"\vec{v}", color=color_v, font_size=24)
        label_v.next_to(arrow_v.get_end(), DR, buff=0.1)

        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(arrow_u), Write(label_u), run_time=1)
        self.play(GrowArrow(arrow_v), Write(label_v), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Revert previous highlight, highlight current
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_uv)

        # Tip-to-tail rule
        # Move arrow_v to end of arrow_u
        new_v_start = arrow_u.get_end()
        new_v_end = plane.c2p(vec_u_coords[0] + vec_v_coords[0], vec_u_coords[1] + vec_v_coords[1])
        
        # Resultant vector u+v
        arrow_uv = Arrow(
            start=origin, 
            end=new_v_end, 
            buff=0, 
            color=color_uv, 
            stroke_width=8
        )
        label_uv = MathTex(r"\vec{u}+\vec{v}", color=color_uv, font_size=24)
        label_uv.next_to(arrow_uv.get_end(), RIGHT, buff=0.1)

        self.play(
            arrow_v.animate.move_to(plane.c2p(vec_u_coords[0] + vec_v_coords[0]/2, vec_u_coords[1] + vec_v_coords[1]/2)),
            label_v.animate.next_to(new_v_end, DR, buff=0.1),
            run_time=2
        )
        self.play(GrowArrow(arrow_uv), Write(label_uv), run_time=1)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Revert previous highlight, highlight current
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_scaled_u)

        # Clear v and uv for scalar multiplication demo
        self.play(FadeOut(arrow_v), FadeOut(label_v), FadeOut(arrow_uv), FadeOut(label_uv))
        
        # Scalar multiplication: 2 * u
        scaled_u_coords = [vec_u_coords[0] * 2, vec_u_coords[1] * 2, 0]
        new_end = plane.c2p(*scaled_u_coords)
        
        label_2u = MathTex(r"2\vec{u}", color=color_scaled_u, font_size=24)
        label_2u.next_to(new_end, UR, buff=0.1)

        self.play(
            arrow_u.animate.scale(2, about_point=origin).set_color(color_scaled_u),
            Transform(label_u, label_2u),
            run_time=2
        )
        self.wait(3)
