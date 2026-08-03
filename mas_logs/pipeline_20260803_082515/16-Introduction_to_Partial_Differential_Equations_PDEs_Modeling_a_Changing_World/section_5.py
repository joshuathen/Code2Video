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
        lecture_lines = [
            "Local constraints create global motion.",
            "Nearby points influence each other's state.",
            "This mimics physical forces like cloth tension.",
            "Simple rules generate complex emergent behavior.",
            "PDEs define the physics of our world."
        ]
        self.setup_layout("Geometric Interpretation: Curvature and Flow", lecture_lines)
        
        # Colors
        COLOR_POINTS = BLUE_B
        COLOR_ARROWS = YELLOW_B
        COLOR_STRESS = "#FF00FF"
        HIGHLIGHT_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create grid at origin first
        n_rows, n_cols = 5, 5
        spacing = 0.5
        
        points = VGroup()
        for i in range(n_rows):
            for j in range(n_cols):
                pos = np.array([j * spacing, -i * spacing, 0])
                dot = Dot(pos, radius=0.06, color=COLOR_POINTS)
                dot.grid_indices = (i, j)
                points.add(dot)
                
        arrows = VGroup()
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                curr_pos = points[idx].get_center()
                # Right neighbor
                if j < n_cols - 1:
                    next_pos = points[idx + 1].get_center()
                    arrows.add(Arrow(curr_pos, next_pos, buff=0.1, color=COLOR_ARROWS, stroke_width=2, max_tip_length_to_length_ratio=0.2))
                # Bottom neighbor
                if i < n_rows - 1:
                    next_pos = points[idx + n_cols].get_center()
                    arrows.add(Arrow(curr_pos, next_pos, buff=0.1, color=COLOR_ARROWS, stroke_width=2, max_tip_length_to_length_ratio=0.2))
        
        grid_group = VGroup(points, arrows)
        # Fix for Issue 30: Positioning grid group
        self.place_in_area(grid_group, 'B3', 'E6', scale_factor=1.2)
        
        self.play(Create(points), run_time=1)
        self.play(Create(arrows), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Highlight interaction: nearby points influence each other
        flash_arrows = arrows.copy().set_color(WHITE).set_stroke(width=4)
        self.play(ShowPassingFlash(flash_arrows, time_width=0.5, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Transition to Mesh for Cloth Simulation
        mesh_lines = VGroup()
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                if j < n_cols - 1:
                    line = Line(points[idx].get_center(), points[idx+1].get_center(), color=COLOR_POINTS, stroke_width=2)
                    line.indices = (idx, idx + 1)
                    mesh_lines.add(line)
                if i < n_rows - 1:
                    line = Line(points[idx].get_center(), points[idx+n_cols].get_center(), color=COLOR_POINTS, stroke_width=2)
                    line.indices = (idx, idx + n_cols)
                    mesh_lines.add(line)
        
        cloth_mesh = VGroup(points, mesh_lines)
        # Fix for Issue 31: Re-center and scale for cloth simulation
        self.place_in_area(cloth_mesh, 'A3', 'F6', scale_factor=1.1)
        
        self.play(
            FadeOut(arrows),
            Create(mesh_lines),
            run_time=1.5
        )
        
        # Setup Animation/Updaters
        time_tracker = ValueTracker(0)
        # Store initial positions for wave center
        for dot in points:
            dot.initial_pos = dot.get_center().copy()
            
        def update_dots(dots):
            t = time_tracker.get_value()
            for dot in dots:
                i, j = dot.grid_indices
                # Local constraints (sin waves)
                offset_x = 0.12 * np.sin(1.0 * i + 1.5 * t)
                offset_y = 0.12 * np.cos(1.0 * j + 1.5 * t)
                dot.move_to(dot.initial_pos + np.array([offset_x, offset_y, 0]))

        def update_lines(lines):
            for line in lines:
                idx1, idx2 = line.indices
                line.set_points_as_corners([points[idx1].get_center(), points[idx2].get_center()])

        points.add_updater(update_dots)
        mesh_lines.add_updater(update_lines)
        
        self.play(time_tracker.animate.set_value(2 * PI), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Simple rules generate complex emergent behavior
        def update_dots_complex(dots):
            t = time_tracker.get_value()
            for dot in dots:
                i, j = dot.grid_indices
                # More complex wave interference pattern
                offset_x = 0.18 * np.sin(1.2 * i + 2.0 * t) + 0.08 * np.sin(2.5 * j + 3.0 * t)
                offset_y = 0.18 * np.cos(1.2 * j + 2.0 * t) + 0.08 * np.cos(2.5 * i + 3.0 * t)
                dot.move_to(dot.initial_pos + np.array([offset_x, offset_y, 0]))

        points.remove_updater(update_dots)
        points.add_updater(update_dots_complex)
        
        self.play(time_tracker.animate.set_value(6 * PI), run_time=5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_STRESS)
        
        def update_lines_with_stress(lines):
            for line in lines:
                idx1, idx2 = line.indices
                p1 = points[idx1].get_center()
                p2 = points[idx2].get_center()
                line.set_points_as_corners([p1, p2])
                
                # Highlight stress: color changes based on distance deviation
                initial_dist = np.linalg.norm(points[idx1].initial_pos - points[idx2].initial_pos)
                current_dist = np.linalg.norm(p1 - p2)
                
                if abs(current_dist - initial_dist) > 0.06:
                    line.set_color(COLOR_STRESS)
                    line.set_stroke(width=3)
                else:
                    line.set_color(COLOR_POINTS)
                    line.set_stroke(width=2)

        mesh_lines.remove_updater(update_lines)
        mesh_lines.add_updater(update_lines_with_stress)
        
        self.play(time_tracker.animate.set_value(10 * PI), run_time=6, rate_func=linear)
        
        # Cleanup updaters at end of section
        points.remove_updater(update_dots_complex)
        mesh_lines.remove_updater(update_lines_with_stress)
        self.wait(2)
