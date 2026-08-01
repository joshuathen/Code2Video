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
        lecture_lines = [
            'We start searching within a large bounding box.',
            'Calculate the winding number along its boundary edges.',
            'If non-zero, subdivide the box into four quadrants.',
            'Recursively search quadrants containing the topological signal.',
            'This process converges precisely onto the hidden root.'
        ]
        self.setup_layout("The Algorithm: Recursive Subdivision", lecture_lines)

        # Colors
        COLOR_BOX = "#FFFFFF"
        COLOR_WINDING = "#FFFF00"
        COLOR_SUBDIV = "#888888"
        COLOR_HIGHLIGHT = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Initial search box - centered in B1-F6 area as per Issue 34
        main_box = Square(side_length=4.0, color=COLOR_BOX, stroke_width=2)
        self.place_in_area(main_box, "B1", "F6", scale_factor=0.8)
        
        self.play(Create(main_box))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Highlight boundary to show winding number calculation
        boundary_path = main_box.copy().set_color(COLOR_WINDING).set_stroke(width=4)
        
        # Boundary arrows for path traversal
        corners = [main_box.get_corner(UL), main_box.get_corner(UR), 
                   main_box.get_corner(DR), main_box.get_corner(DL)]
        arrow_group = VGroup(*[
            Arrow(start=corners[i], end=corners[(i+1)%4], color=COLOR_WINDING, buff=0, tip_length=0.2)
            for i in range(4)
        ])

        self.play(Create(boundary_path))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrow_group], lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Subdivision lines
        box_center = main_box.get_center()
        box_side = main_box.side_length
        v_line = Line(box_center + UP * box_side/2, box_center + DOWN * box_side/2, color=COLOR_SUBDIV)
        h_line = Line(box_center + LEFT * box_side/2, box_center + RIGHT * box_side/2, color=COLOR_SUBDIV)
        
        self.play(FadeOut(arrow_group), FadeOut(boundary_path))
        self.play(Create(v_line), Create(h_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Highlight the quadrant with the "signal" (top-right)
        tr_quad_box = Square(side_length=box_side/2, color=COLOR_HIGHLIGHT, stroke_width=4)
        # Position using grid-relative logic to avoid manual move_to
        self.place_in_area(tr_quad_box, "B4", "D6", scale_factor=0.8)
        
        # Subdivide that quadrant further
        tr_center = tr_quad_box.get_center()
        tr_side = tr_quad_box.side_length
        v_line2 = Line(tr_center + UP * tr_side/2, tr_center + DOWN * tr_side/2, color=COLOR_SUBDIV, stroke_width=1)
        h_line2 = Line(tr_center + LEFT * tr_side/2, tr_center + RIGHT * tr_side/2, color=COLOR_SUBDIV, stroke_width=1)
        
        self.play(Create(tr_quad_box))
        self.play(Create(v_line2), Create(h_line2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Load pinwheel asset (Issue 24)
        pinwheel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/pinwheel.svg")
        # Place pinwheel in the target subdivision (converged root location)
        # Using a grid point within the previous quadrant's area
        self.place_at_grid(pinwheel, "C5", scale_factor=0.4)
        
        # Smaller box representing the final convergence
        converged_box = Square(side_length=tr_side/2, color=COLOR_HIGHLIGHT, stroke_width=2)
        converged_box.move_to(pinwheel.get_center())
        
        self.play(ReplacementTransform(tr_quad_box, converged_box))
        self.play(FadeIn(pinwheel))
        self.play(pinwheel.animate.scale(1.5), converged_box.animate.scale(0.8).set_stroke(width=1))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
