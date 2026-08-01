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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Configuration Space: Mapping Pairs",
            [
                "We map every pair of points to 3D space.",
                "Height represents the distance between the points.",
                "The base coordinates represent their unique midpoint.",
                "Moving points along the loop creates a surface.",
                "This \"configuration space\" captures all possible pairings."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Line 1 corresponds to mapping points (Red)
        self.lecture[0].set_color(RED)
        
        # Draw a loop (white ellipse) in the bottom-right area (Rows E and F)
        loop = Ellipse(width=3.5, height=1.2, color=WHITE)
        self.place_in_area(loop, "E2", "F5")
        self.play(Create(loop))

        # ValueTrackers for moving points along the loop (proportion 0 to 1)
        p1_tracker = ValueTracker(0.1)
        p2_tracker = ValueTracker(0.5)

        # Define points and connector line (always_redraw keeps them updated)
        p1 = always_redraw(lambda: Dot(loop.point_from_proportion(p1_tracker.get_value() % 1), color=RED))
        p2 = always_redraw(lambda: Dot(loop.point_from_proportion(p2_tracker.get_value() % 1), color=RED))
        connector = always_redraw(lambda: Line(p1.get_center(), p2.get_center(), color=RED, stroke_width=2))

        self.play(FadeIn(p1), FadeIn(p2), Create(connector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2 corresponds to Height (Green)
        self.lecture[1].set_color(GREEN)
        
        # Function to compute midpoint and scaled distance for height
        def get_height_params():
            pos1 = p1.get_center()
            pos2 = p2.get_center()
            mid = (pos1 + pos2) / 2
            dist = np.linalg.norm(pos1 - pos2)
            # Scale factor 0.7 for visual balance in the 2D scene
            return mid, dist * 0.7

        # Vertical green vector representing height = distance
        height_vector = always_redraw(lambda: Arrow(
            start=get_height_params()[0],
            end=get_height_params()[0] + UP * get_height_params()[1],
            color=GREEN,
            buff=0,
            stroke_width=4
        ))
        
        self.play(GrowArrow(height_vector))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3 corresponds to Midpoint (White)
        self.lecture[2].set_color(WHITE)
        
        midpoint_dot = always_redraw(lambda: Dot(get_height_params()[0], color=WHITE, radius=0.04))
        m_label = Text("M", font_size=18, color=WHITE)
        # Position label in the grid near the loop plane
        self.place_at_grid(m_label, "F4")
        
        self.play(FadeIn(midpoint_dot), Write(m_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4 corresponds to the resulting surface/canopy (Blue)
        self.lecture[3].set_color(BLUE)
        
        # Tip of the vector traces the canopy
        tip_trace = TracedPath(height_vector.get_end, stroke_color=BLUE, stroke_width=3)
        self.add(tip_trace)

        # Animate points moving along the loop to sweep the space
        self.play(
            p1_tracker.animate.set_value(1.1),
            p2_tracker.animate.set_value(0.9),
            run_time=4,
            rate_func=linear
        )
        
        # Visualize the "canopy" (configuration space) as a blue translucent shape
        # Located in the area above the loop (Rows B through E)
        canopy_shape = Ellipse(width=3.6, height=2.4, color=BLUE, fill_opacity=0.3)
        self.place_in_area(canopy_shape, "B2", "E5")
        
        self.play(FadeIn(canopy_shape))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: Labeling the whole configuration space
        self.lecture[4].set_color(WHITE)
        
        config_label = Text("Configuration Space", font_size=20, color=WHITE)
        # Placing label at the top of the right-hand animation area
        self.place_at_grid(config_label, "A3")
        
        self.play(Write(config_label))
        self.wait(2)
