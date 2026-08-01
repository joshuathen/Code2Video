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
        # Initial layout setup
        title = "The Coastline Paradox: Why Standard Rulers Fail"
        lines = [
            "How long is the coastline of Britain?",
            "A seagull's 1-kilometer ruler gives a fixed length.",
            "A crab's tiny ruler finds many more details.",
            "Smaller rulers make the coastline grow much longer.",
            "Standard geometry fails to measure this infinite roughness."
        ]
        self.setup_layout(title, lines)

        # Define basic points for a simplified Britain outline (centered at 0,0)
        # 11 points to create 10 segments
        britain_pts = [
            [0, 1.8, 0], [0.6, 1.4, 0], [0.4, 0.6, 0], [0.8, 0.1, 0],
            [0.6, -0.8, 0], [0.2, -1.8, 0], [-0.4, -1.6, 0], [-0.8, -0.1, 0],
            [-0.4, 0.7, 0], [-0.7, 1.4, 0], [0, 1.8, 0]
        ]

        # === Animation for Lecture Line 1 ===
        # Draw a simplified blue (#ADD8E6) outline of Britain in the center.
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        britain_outline = VMobject(color="#ADD8E6")
        britain_outline.set_points_as_corners(britain_pts)
        self.place_in_area(britain_outline, 'A1', 'F6', scale_factor=1.0)
        
        self.play(Create(britain_outline), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A series of 10 white (#FFFFFF) line segments (1km rulers) trace the outline.
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        # Use get_anchors() to retrieve the vertices of the VMobject after transformation
        placed_pts = britain_outline.get_anchors()
        rulers_1km = VGroup(*[
            Line(placed_pts[i], placed_pts[i+1], color="#FFFFFF", stroke_width=4)
            for i in range(len(placed_pts)-1)
        ])
        
        self.play(Create(rulers_1km), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Zoom into the west coast; replace the 1km segments with 100 tiny yellow (#FFFF00) segments.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # We focus on the "west coast" segment (around point 7 to 8 in placed_pts)
        focus_point = placed_pts[7]
        zoom_factor = 4
        
        # Prepare zoomed-in version of the outline
        zoomed_britain = britain_outline.copy().scale(zoom_factor, about_point=focus_point)
        # Shift everything so focus_point is at the center of the grid area
        grid_center = (self.grid['A1'] + self.grid['F6']) / 2
        shift_vec = grid_center - focus_point
        zoomed_britain.shift(shift_vec)

        # Create 100 tiny yellow segments (simulating 1cm rulers) along the zoomed west coast
        # Using get_anchors() to extract points from the transformed zoomed object
        z_start = zoomed_britain.get_anchors()[7]
        z_end = zoomed_britain.get_anchors()[8]
        
        yellow_pts = []
        num_yellow = 50 # Using 50 for visual clarity
        np.random.seed(123)
        for i in range(num_yellow + 1):
            alpha = i / num_yellow
            p = (1-alpha)*z_start + alpha*z_end
            if 0 < i < num_yellow:
                # Add some perpendicular jitter to show "found details"
                perp = np.array([-(z_end[1]-z_start[1]), z_end[0]-z_start[0], 0])
                if np.linalg.norm(perp) > 0:
                    perp = perp / np.linalg.norm(perp)
                p += perp * np.random.uniform(-0.15, 0.15)
            yellow_pts.append(p)
            
        rulers_1cm = VGroup(*[
            Line(yellow_pts[i], yellow_pts[i+1], color="#FFFF00", stroke_width=2)
            for i in range(len(yellow_pts)-1)
        ])

        self.play(
            ReplacementTransform(britain_outline, zoomed_britain),
            FadeOut(rulers_1km),
            run_time=2
        )
        self.play(Create(rulers_1cm), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Text 'Length: Increasing...' appears in green (#00FF00)
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        length_text = Text("Length: Increasing...", font_size=24, color="#00FF00")
        self.place_at_grid(length_text, 'E5')
        
        self.play(Write(length_text))
        
        # Jitter the yellow lines to suggest increasing complexity
        for _ in range(2):
            self.play(rulers_1cm.animate.shift(UP*0.05), run_time=0.2)
            self.play(rulers_1cm.animate.shift(DOWN*0.05), run_time=0.2)
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Replace outline with a high-detail white (#FFFFFF) 'infinite' curve.
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Create a full-view very detailed curve
        high_detail_pts = []
        for i in range(len(britain_pts)-1):
            s = np.array(britain_pts[i])
            e = np.array(britain_pts[i+1])
            for k in range(8): # 8 sub-segments per side
                alpha = k/8
                p = (1-alpha)*s + alpha*e
                if k > 0:
                    p += np.array([np.random.uniform(-0.04, 0.04), np.random.uniform(-0.04, 0.04), 0])
                high_detail_pts.append(p)
        high_detail_pts.append(britain_pts[-1])
        
        final_curve = VMobject(color="#FFFFFF", stroke_width=1.5)
        final_curve.set_points_as_corners(high_detail_pts)
        self.place_in_area(final_curve, 'A1', 'F6', scale_factor=1.0)

        self.play(
            FadeOut(zoomed_britain),
            FadeOut(rulers_1cm),
            FadeOut(length_text),
            Create(final_curve),
            run_time=2
        )
        self.wait(2)
