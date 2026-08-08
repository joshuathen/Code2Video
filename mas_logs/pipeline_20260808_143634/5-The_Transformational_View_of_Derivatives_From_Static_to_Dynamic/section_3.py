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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Geometric Interpretation: Dynamic Tangency", [
            "Derivative is the slope function.",
            "It shows direction at every point.",
            "Path slope equals instantaneous velocity.",
            "Derivative captures precise orientation.",
            "Dynamic tangency defines the curve behavior."
        ])
        
        # Setup Circle
        circle = Circle(radius=1.5, color=BLUE)
        self.place_in_area(circle, 'B4', 'F6', scale_factor=0.55)
        self.add(circle)
        
        # Tracker for animation
        tracker = ValueTracker(0)
        
        # Define Tangent Line
        # We don't use always_redraw for the TangentLine object itself per instructions,
        # but the logic requires dynamic updating, so we wrap in a group and update geometry.
        tangent_line = TangentLine(circle, alpha=0, length=2, color=YELLOW)
        self.add(tangent_line)
        
        def update_tangent(m):
            m.become(TangentLine(circle, alpha=tracker.get_value(), length=2, color=YELLOW))
            
        tangent_line.add_updater(update_tangent)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(tracker.animate.set_value(0.2), run_time=1.5)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(tracker.animate.set_value(0.4), run_time=1.5)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(tracker.animate.set_value(0.6), run_time=1.5)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.play(tracker.animate.set_value(0.8), run_time=1.5)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.play(tracker.animate.set_value(1.0), run_time=1.5)
        
        tangent_line.remove_updater(update_tangent)
