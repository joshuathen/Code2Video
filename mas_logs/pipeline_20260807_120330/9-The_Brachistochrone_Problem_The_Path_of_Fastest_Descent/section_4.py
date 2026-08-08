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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The 'Aha!' Moment: Mechanical Light", [
            "Imagine the vertical space as many thin layers.",
            "The marble speeds up as it crosses each layer.",
            "This is exactly how light behaves in varying media.",
            "The fastest descent follows the path of light.",
            "This link reveals the optimal descent curve."
        ])

        # === Animation for Lecture Line 1 ===
        # Create horizontal layers relative to origin
        layers_list = [Line(LEFT * 2.5, RIGHT * 2.5, color="#333333", stroke_width=2) for _ in range(7)]
        layers_group = VGroup(*layers_list).arrange(DOWN, buff=1.0)
        # Apply fix for Issue 29: Use grid-constrained area for horizontal lines
        self.place_in_area(layers_group, 'A2', 'F6', scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color("#333333"))
        self.play(Create(layers_group), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create the marble's path and segments. 
        # To satisfy Issue 31, we define them in a group and place them in the grid area.
        local_layers = VGroup(*[Line(LEFT * 2.5, RIGHT * 2.5) for _ in range(7)]).arrange(DOWN, buff=1.0)
        # Points based on the local coordinate system of the layers group
        local_pts = [local_layers[i].point_from_proportion(0.1 + 0.12*i + 0.005*i*i) for i in range(7)]
        local_segs = VGroup(*[
            Line(local_pts[i], local_pts[i+1], color="#00FF00", stroke_width=4) 
            for i in range(6)
        ])
        
        # Apply fix for Issue 31: Place path group in area
        self.place_in_area(local_segs, 'A2', 'F6', scale_factor=0.8)
        
        # The marble should start at the beginning of the path
        marble = Dot(local_segs[0].get_start(), color=WHITE, radius=0.08)
        
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(Create(local_segs), run_time=2)
        
        self.add(marble)
        # Marble moves along the segmented path with increasing speed
        for i, seg in enumerate(local_segs):
            # Simulation of acceleration: duration for each layer decreases
            duration = 0.6 * (0.8 ** i)
            self.play(MoveAlongPath(marble, seg), run_time=duration, rate_func=linear)
            
        self.play(FadeOut(marble))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Apply fix for Issue 30: Place velocity labels in grid area (Column 1)
        v_labels = VGroup(*[
            MathTex(f"v_{i+1}", color="#00FF00", font_size=28) for i in range(6)
        ]).arrange(DOWN, buff=0.8)
        self.place_in_area(v_labels, 'A1', 'F1', scale_factor=0.6)
        
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(Write(v_labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Morph segmented path to a smooth curve representing the light path
        # Extract points from the positioned segments to build the curve
        smooth_points = [local_segs[0].get_start()] + [s.get_end() for s in local_segs]
        smooth_curve = VMobject(color="#FF00FF", stroke_width=4)
        smooth_curve.set_points_as_corners(smooth_points).make_smooth()
        
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(
            FadeOut(v_labels),
            Transform(local_segs, smooth_curve)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        # Highlighting the curve to finalize the link between descent and light
        self.play(local_segs.animate.set_stroke(width=8), run_time=0.4)
        self.play(local_segs.animate.set_stroke(width=4), run_time=0.4)
        self.wait(2)
