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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Key Property: Locality Preservation"
        lecture_lines = [
            "Nearby points on the line remain nearby in space.",
            "This locality preservation is useful for data clustering.",
            "The curve cleans one region before moving to another."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define colors for each animation stage
        COLOR_1 = "#FFA500" # Orange
        COLOR_2 = "#00FFFF" # Cyan
        COLOR_3 = "#FFFF00" # Yellow
        
        # --- Prepare Objects ---\n        
        # 1. 1D Line Representation
        # Line spans part of the grid area
        line_main = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        # Create 16 dots representing mapped points on the 1D line
        line_dots = VGroup(*[Dot(line_main.point_from_proportion(i/15), radius=0.06, color=WHITE) for i in range(16)])
        line_group = VGroup(line_main, line_dots)
        # Issue 39: Move line_group to row B to avoid crowding title
        self.place_in_area(line_group, "B1", "B6", scale_factor=1.0)
        
        # 2. 2D Hilbert Curve (Order 2)
        # Defining points in the standard Hilbert order for N=2
        pts = [
            [0,0,0], [1,0,0], [1,1,0], [0,1,0], # Quadrant 0 (Lower-Left)
            [0,2,0], [0,3,0], [1,3,0], [1,2,0], # Quadrant 1 (Upper-Left)
            [2,2,0], [2,3,0], [3,3,0], [3,2,0], # Quadrant 2 (Upper-Right)
            [3,1,0], [2,1,0], [2,0,0], [3,0,0]  # Quadrant 3 (Lower-Right)
        ]
        h_dots = VGroup(*[Dot(np.array(p), radius=0.08, color=WHITE) for p in pts])
        h_curve = VMobject(color=WHITE, stroke_width=4)
        h_curve.set_points_as_corners([d.get_center() for d in h_dots])
        
        h_group = VGroup(h_curve, h_dots)
        # Issue 40: Scale down h_group slightly to 0.8
        self.place_in_area(h_group, "C1", "F6", scale_factor=0.8)
        
        # Add primary objects to scene
        self.add(line_group, h_group)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Highlight index 6 and 7 (randomly chosen nearby points in the UL quadrant)
        idx_a, idx_b = 6, 7
        highlight_dots_1d = VGroup(line_dots[idx_a], line_dots[idx_b])
        highlight_dots_2d = VGroup(h_dots[idx_a], h_dots[idx_b])
        
        self.play(
            highlight_dots_1d.animate.set_color(COLOR_1).scale(1.5),
            highlight_dots_2d.animate.set_color(COLOR_1).scale(1.5),
            run_time=1
        )
        self.play(Flash(highlight_dots_2d, color=COLOR_1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Revert colors of Line 1
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            highlight_dots_1d.animate.set_color(WHITE).scale(1/1.5),
            highlight_dots_2d.animate.set_color(WHITE).scale(1/1.5),
            self.lecture[1].animate.set_color(COLOR_2)
        )
        
        # Show a "Cluster": sequence 8, 9, 10, 11 (corresponds to the Upper-Right quadrant)
        cluster_indices = [8, 9, 10, 11]
        cluster_1d = VGroup(*[line_dots[i] for i in cluster_indices])
        cluster_2d_dots = VGroup(*[h_dots[i] for i in cluster_indices])
        # Draw a specific path highlighting this sequence
        cluster_2d_path = VMobject(color=COLOR_2, stroke_width=6).set_points_as_corners([h_dots[i].get_center() for i in cluster_indices])

        self.play(
            cluster_1d.animate.set_color(COLOR_2),
            cluster_2d_dots.animate.set_color(COLOR_2),
            Create(cluster_2d_path),
            run_time=1.5
        )
        self.play(Indicate(cluster_2d_dots))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Clean up previous highlights
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            cluster_1d.animate.set_color(WHITE),
            cluster_2d_dots.animate.set_color(WHITE),
            FadeOut(cluster_2d_path),
            self.lecture[2].animate.set_color(COLOR_3)
        )
        
        # Animate quadrants sequentially to show regional completion
        h_curve.set_stroke(opacity=0.2) # Dim the background curve
        quadrant_indices = [
            [0, 1, 2, 3],    # LL
            [4, 5, 6, 7],    # UL
            [8, 9, 10, 11],  # UR
            [12, 13, 14, 15] # LR
        ]
        
        for i, q_indices in enumerate(quadrant_indices):
            # Define path for this specific quadrant
            q_pts = [h_dots[j].get_center() for j in q_indices]
            segment_path = VMobject(color=COLOR_3, stroke_width=6)
            segment_path.set_points_as_corners(q_pts)
            
            # Animate the connection to the previous quadrant if not the first
            if i > 0:
                prev_end = h_dots[quadrant_indices[i-1][-1]].get_center()
                curr_start = h_dots[q_indices[0]].get_center()
                connector = Line(prev_end, curr_start, color=COLOR_3, stroke_width=6)
                self.play(Create(connector), run_time=0.3)
            
            # Animate the quadrant itself
            self.play(Create(segment_path), run_time=1.2)
            self.wait(0.2)
            
        self.wait(3)
