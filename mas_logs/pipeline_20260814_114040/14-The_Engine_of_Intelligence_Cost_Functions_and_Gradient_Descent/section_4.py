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
        self.setup_layout("The Learning Rate: Setting the Pace", [
            "- Learning rate controls our step size.", 
            "- Large steps might jump over the minimum.", 
            "- Small steps ensure steady, careful progress."
        ])
        
        # Fixing issue 30: position valley higher
        valley = FunctionGraph(lambda x: 0.5 * (x**2), x_range=[-3, 3], color=BLUE)
        self.place_in_area(valley, 'A3', 'C6', scale_factor=0.8)
        
        # Terminal anchor (B007)
        target = Dot(color=GREEN).move_to(valley.get_bottom())
        self.add(valley, target)

        # Hiker position
        hiker_x = ValueTracker(-2.5)
        hiker = Dot(color=YELLOW)
        # Fix issue 31: use place_at_grid-like logic but keep it centered on valley
        hiker.add_updater(lambda d: d.move_to(valley.point_from_proportion((hiker_x.get_value() + 3) / 6)))
        self.add(hiker)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(4) # B029

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        # Large step (overshoot)
        self.play(hiker_x.animate.set_value(0.5), run_time=1.0)
        self.play(hiker_x.animate.set_value(2.0), run_time=1.0) # OVERSHOOT
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        # Small steps (convergence)
        hiker_x.set_value(-2.5)
        for _ in range(5):
            self.play(hiker_x.animate.set_value(hiker_x.get_value() + 0.6), run_time=0.4)
        self.wait(3)
