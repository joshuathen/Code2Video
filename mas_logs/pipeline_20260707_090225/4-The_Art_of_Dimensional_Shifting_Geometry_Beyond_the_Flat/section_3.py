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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            'Can six matchsticks form four equilateral triangles?',
            'We can easily build one triangle in 2D.',
            'But the remaining sticks fail to form more.',
            'Lift the sticks into the third dimension instead.',
            'Now, a 3D tetrahedron creates four perfect triangles.'
        ]
        self.setup_layout("Puzzle 1: The Matchstick Triangle (The 2D to 3D Leap)", lecture_lines)

        # Colors
        STICK_BROWN = "#A52A2A"
        TRI_YELLOW = "#FFFF00"
        FACE_GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # 'Can six matchsticks form four equilateral triangles?'
        self.lecture[0].set_color(STICK_BROWN)
        
        # Create 6 brown matchstick segments
        sticks = VGroup(*[Line(ORIGIN, RIGHT * 0.8, color=STICK_BROWN, stroke_width=6) for _ in range(6)])
        # Scatter them randomly
        for i, s in enumerate(sticks):
            s.rotate(i * 0.9)
            s.shift(UP * (i%3 - 1) * 0.5 + RIGHT * (i//3 - 0.5) * 0.5)
        
        # Fix Issue 34: Position scattered matchsticks
        self.place_in_area(sticks, 'B2', 'E5', scale_factor=0.7)
        
        self.play(LaggedStartMap(Create, sticks, lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'We can easily build one triangle in 2D.'
        self.lecture[1].set_color(TRI_YELLOW)
        
        # Define equilateral triangle geometry at origin first
        side_len = 1.2
        height = side_len * np.sqrt(3) / 2
        v0 = np.array([0, 2*height/3, 0])
        v1 = np.array([-side_len/2, -height/3, 0])
        v2 = np.array([side_len/2, -height/3, 0])
        
        target_tri_2d = VGroup(
            Line(v1, v2),
            Line(v2, v0),
            Line(v0, v1)
        )
        # Fix Issue 35: Center triangle at C4
        self.place_at_grid(target_tri_2d, 'C4', scale_factor=0.9)
        
        # Move first 3 sticks to form yellow triangle
        self.play(
            sticks[0].animate.put_start_and_end_on(target_tri_2d[0].get_start(), target_tri_2d[0].get_end()).set_color(TRI_YELLOW),
            sticks[1].animate.put_start_and_end_on(target_tri_2d[1].get_start(), target_tri_2d[1].get_end()).set_color(TRI_YELLOW),
            sticks[2].animate.put_start_and_end_on(target_tri_2d[2].get_start(), target_tri_2d[2].get_end()).set_color(TRI_YELLOW),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'But the remaining sticks fail to form more.'
        self.lecture[2].set_color(STICK_BROWN)
        
        # Move other sticks to overlapping/failed positions relative to triangle vertices
        v_base = target_tri_2d[0].get_start()
        v_apex_2d = target_tri_2d[1].get_end()
        self.play(
            sticks[3].animate.put_start_and_end_on(v_base + LEFT*0.4, v_base + DOWN*0.7),
            sticks[4].animate.put_start_and_end_on(v_apex_2d + UP*0.2, v_apex_2d + RIGHT*0.5 + UP*0.5),
            sticks[5].animate.put_start_and_end_on(target_tri_2d[0].get_end() + RIGHT*0.3, target_tri_2d[0].get_end() + DOWN*0.6),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'Lift the sticks into the third dimension instead.'
        self.lecture[3].set_color(STICK_BROWN)
        
        # Define Tetrahedron Projection (Isometric View)
        VA = np.array([0, 1.2, 0])
        VB = np.array([-1.0, -0.6, 0])
        VC = np.array([1.0, -0.6, 0])
        VD = np.array([0, -0.1, 0])
        
        tet_edges_target = VGroup(
            Line(VB, VC), Line(VC, VD), Line(VD, VB),
            Line(VA, VB), Line(VA, VC), Line(VA, VD)
        )
        tet_faces = VGroup(
            Polygon(VB, VC, VD, fill_opacity=0.4, fill_color=FACE_GOLD, stroke_width=0), # base
            Polygon(VA, VB, VD, fill_opacity=0.5, fill_color=FACE_GOLD, stroke_width=0), # left
            Polygon(VA, VC, VD, fill_opacity=0.3, fill_color=FACE_GOLD, stroke_width=0), # right
            Polygon(VA, VB, VC, fill_opacity=0.6, fill_color=FACE_GOLD, stroke_width=0)  # front
        )
        tet_nums = VGroup(
            Text("1", font_size=24).move_to((VB + VC + VD)/3),
            Text("2", font_size=24).move_to((VA + VB + VD)/3),
            Text("3", font_size=24).move_to((VA + VC + VD)/3),
            Text("4", font_size=24).move_to((VA + VB + VC)/3)
        )
        
        # Fix Issue 36: Group tetrahedron for grid anchoring
        tetrahedron_group = VGroup(tet_edges_target, tet_faces, tet_nums)
        self.place_in_area(tetrahedron_group, 'B3', 'E5', scale_factor=0.8)
        
        # Animate all 6 sticks to their tetrahedron edge positions
        self.play(
            *[sticks[i].animate.put_start_and_end_on(tet_edges_target[i].get_start(), tet_edges_target[i].get_end()).set_color(TRI_YELLOW if i < 3 else STICK_BROWN) for i in range(6)],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'Now, a 3D tetrahedron creates four perfect triangles.'
        self.lecture[4].set_color(FACE_GOLD)
        
        # Reveal gold faces and numbers
        self.play(FadeIn(tet_faces), FadeIn(tet_nums))
        self.play(
            tet_faces.animate.set_fill(opacity=0.8),
            tet_nums.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
