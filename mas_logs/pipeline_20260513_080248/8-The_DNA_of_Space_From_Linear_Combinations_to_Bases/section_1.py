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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisites: Vectors as Instructions", 
            [
                'Think of vectors as instructions for movement.', 
                'Vector v moves us east and north.', 
                'Scaling v stretches or shrinks this movement.'
            ]
        )
        
        # Define common colors for alignment
        COLOR_V = "#00FFFF"
        COLOR_COMPONENTS = "#FFFF00"
        COLOR_SCALING = "#FF00FF"

        # Initialize the coordinate system
        # Use a NumberPlane that fits comfortably in the right-side grid area
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 5, 1],
            x_length=4,
            y_length=5,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        self.place_in_area(plane, 'A1', 'F6')

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_V))
        
        # Define vector v = [1, 2]
        v_start = plane.c2p(0, 0)
        v_end = plane.c2p(1, 2)
        v_arrow = Arrow(v_start, v_end, buff=0, color=COLOR_V, stroke_width=6)
        
        # Label for the vector
        v_label = Text("Movement", font_size=20, color=COLOR_V)
        # Position label near the vector tip using the grid system
        self.place_at_grid(v_label, 'B5') 
        
        self.play(Create(plane), GrowArrow(v_arrow))
        self.play(FadeIn(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_COMPONENTS))
        
        # Asset: small dot icon
        dot_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dot.svg")
        dot_asset.set_color(WHITE)
        dot_asset.scale(0.15)
        dot_asset.move_to(plane.c2p(0, 0))
        
        # Dashed components
        h_comp = DashedLine(plane.c2p(0, 0), plane.c2p(1, 0), color=COLOR_COMPONENTS)
        v_comp = DashedLine(plane.c2p(1, 0), plane.c2p(1, 2), color=COLOR_COMPONENTS)
        
        # Labels for components
        h_label = Text("+1 East", font_size=16, color=COLOR_COMPONENTS)
        v_label_comp = Text("+2 North", font_size=16, color=COLOR_COMPONENTS)
        self.place_at_grid(h_label, 'D4')
        self.place_at_grid(v_label_comp, 'C5')

        self.play(FadeIn(dot_asset))
        # Move dot along the vector path while showing components
        self.play(
            dot_asset.animate.move_to(plane.c2p(1, 2)),
            Create(h_comp),
            Create(v_comp),
            FadeIn(h_label),
            FadeIn(v_label_comp),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_SCALING))
        
        # Scaling vectors: 2v and 0.5v
        v2_end = plane.c2p(2, 4)
        v_half_end = plane.c2p(0.5, 1)
        
        v2_arrow = Arrow(plane.c2p(0, 0), v2_end, buff=0, color=COLOR_SCALING, stroke_width=6)
        v_half_arrow = Arrow(plane.c2p(0, 0), v_half_end, buff=0, color=COLOR_SCALING, stroke_width=6)
        
        # Labels for scaling
        label_2v = Text("2v (Stretch)", font_size=20, color=COLOR_SCALING)
        label_halfv = Text("0.5v (Shrink)", font_size=20, color=COLOR_SCALING)
        self.place_at_grid(label_2v, 'A5')
        self.place_at_grid(label_halfv, 'C4')

        # Transition: Stretch to 2v
        self.play(
            ReplacementTransform(v_arrow, v2_arrow),
            FadeOut(v_label),
            FadeIn(label_2v),
            dot_asset.animate.move_to(v2_end)
        )
        self.wait(1)
        
        # Transition: Shrink to 0.5v
        self.play(
            ReplacementTransform(v2_arrow, v_half_arrow),
            FadeOut(label_2v),
            FadeIn(label_halfv),
            dot_asset.animate.move_to(v_half_end)
        )
        self.wait(2)
