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
        # Initialize the scene layout with updated script
        self.setup_layout(
            "Topology Twist: The Möbius Strip", 
            [
                'Start with a simple, flat rectangular strip.', 
                'Add a half-twist to the center of the strip.', 
                'Joining the ends creates the famous Möbius loop.', 
                'Follow the path along its single continuous surface.', 
                'One journey visits both sides without crossing an edge.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Load and place the strip asset
        strip = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/strip.svg")
        strip.set_color(WHITE)
        # Issue 40 fix: Move to C2-D5
        self.place_in_area(strip, "C2", "D5")
        
        self.lecture[0].set_color(WHITE)
        self.play(DrawBorderThenFill(strip))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a "twisted" representation (narrows in center)
        # We'll transform the asset into a polygon that simulates a twist
        w = strip.width
        h = strip.height
        c = strip.get_center()
        
        twisted_points = [
            c + np.array([-w/2, h/2, 0]),   # UL
            c + np.array([0, -h/4, 0]),     # Twist center dip
            c + np.array([w/2, h/2, 0]),    # UR
            c + np.array([w/2, -h/2, 0]),   # DR
            c + np.array([0, h/4, 0]),      # Twist center rise
            c + np.array([-w/2, -h/2, 0]),  # DL
        ]
        twisted_strip = Polygon(*twisted_points, color=WHITE, fill_opacity=1.0)
        twisted_strip.set_fill(color=["#FFFFFF", "#888888", "#FFFFFF"])
        
        self.lecture[1].set_color(WHITE)
        self.play(Transform(strip, twisted_strip))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Create the blue Möbius loop
        mobius_path = ParametricFunction(
            lambda t: np.array([
                2.2 * np.cos(t),
                1.0 * np.sin(2 * t),
                0
            ]),
            t_range=[0, TAU],
            color="#00BFFF"
        )
        
        mobius_loop_visual = mobius_path.copy().set_stroke(width=16)
        # Issue 41 fix: Move to B2-E6
        self.place_in_area(mobius_loop_visual, "B2", "E6")
        
        # Sync the path used for animation with the scaled visual object
        mobius_path.match_points(mobius_loop_visual)

        self.lecture[2].set_color("#00BFFF")
        self.play(
            Transform(strip, mobius_loop_visual),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Orange dot begins traversal
        dot = Dot(color="#FF4500", radius=0.15)
        dot.move_to(mobius_path.get_start())
        
        self.lecture[3].set_color("#FF4500")
        self.play(FadeIn(dot))
        
        # First half of the journey
        self.play(
            MoveAlongPath(dot, mobius_path),
            run_time=3,
            rate_func=lambda t: t * 0.5  # Move only halfway
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Second half of the journey
        self.lecture[4].set_color("#FF4500")
        self.play(
            MoveAlongPath(dot, mobius_path),
            run_time=3,
            rate_func=lambda t: 0.5 + t * 0.5 # Finish from halfway
        )
        
        self.play(FadeOut(dot))
        self.wait(2)
