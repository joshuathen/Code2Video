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
        title = "The Hook: The Speeding Falcon"
        lines = [
            "Imagine a falcon diving through the air.",
            "We can calculate its average speed easily.",
            "But how fast is it going right now?"
        ]
        self.setup_layout(title, lines)

        # Adjusted Function for the parabolic path to start at B1 and end at F6
        # Start: B1 (0.5, 1.2), End: F6 (5.5, -2.8)
        # Formula: y = 1.2 - 0.16 * (x - 0.5)^2
        def path_func(x):
            return 1.2 - 0.16 * (x - 0.5)**2

        path_curve = FunctionGraph(
            path_func,
            x_range=[0.5, 5.5],
            color=BLUE_E
        )

        # Falcon silhouette representation
        # Using a Triangle as a placeholder for the falcon asset
        falcon = Triangle(color="#3399FF", fill_opacity=1)
        # Fix for Issue 22 & 23: Move to B1 and scale up
        self.place_at_grid(falcon, "B1", scale_factor=1.5)
        falcon.rotate(-PI/2) # Pointing down-right roughly
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        self.add(path_curve, falcon)
        self.play(FadeIn(path_curve), FadeIn(falcon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Falcon dives along path using ValueTracker and updater
        falcon_pos_tracker = ValueTracker(0.5)
        falcon.add_updater(lambda m: m.move_to(np.array([
            falcon_pos_tracker.get_value(),
            path_func(falcon_pos_tracker.get_value()),
            0
        ])))
        
        # Points for average speed
        p1_x = 1.5
        p2_x = 4.5
        p1 = Dot(np.array([p1_x, path_func(p1_x), 0]), color="#FFD700")
        p2 = Dot(np.array([p2_x, path_func(p2_x), 0]), color="#FFD700")
        
        # Secant line (Dashed white)
        secant_line = DashedLine(p1.get_center(), p2.get_center(), color=WHITE)
        
        self.play(
            falcon_pos_tracker.animate.set_value(5.5),
            run_time=3,
            rate_func=linear
        )
        self.play(Create(p1), Create(p2))
        self.play(Create(secant_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # ValueTracker for p2's relative position to p1 to demonstrate limit
        dx_tracker = ValueTracker(p2_x - p1_x)
        
        # Updater for point p2 to slide towards p1
        p2.add_updater(lambda m: m.move_to(np.array([
            p1_x + dx_tracker.get_value(),
            path_func(p1_x + dx_tracker.get_value()),
            0
        ])))
        
        # Persistent tangent/secant line with updater for performance
        live_line = Line(p1.get_center(), p2.get_center())
        
        def update_line(line):
            start_x = p1_x
            end_x = p1_x + dx_tracker.get_value()
            p_start = np.array([start_x, path_func(start_x), 0])
            p_end = np.array([end_x, path_func(end_x), 0])
            
            # Transition from secant to tangent
            if abs(end_x - start_x) < 0.05:
                # Use derivative for precise tangent when points overlap
                # dy/dx = -0.32 * (x - 0.5)
                slope = -0.32 * (start_x - 0.5)
                direction = np.array([1, slope, 0])
                direction = direction / np.linalg.norm(direction)
                line.set_points_by_ends(p_start - direction * 1.5, p_start + direction * 1.5)
                line.set_color("#FFD700")
            else:
                # Secant line direction
                direction = p_end - p_start
                direction = direction / np.linalg.norm(direction)
                line.set_points_by_ends(p_start - direction * 1.0, p_end + direction * 1.0)
                if dx_tracker.get_value() > 0.1:
                    line.set_color(WHITE)
                else:
                    line.set_color("#FFD700")

        live_line.add_updater(update_line)
        self.remove(secant_line)
        self.add(live_line)
        
        # Move falcon back to focus point p1 and collapse the interval
        self.play(
            falcon_pos_tracker.animate.set_value(p1_x),
            dx_tracker.animate.set_value(0.001),
            run_time=4
        )
        
        self.wait(2)
