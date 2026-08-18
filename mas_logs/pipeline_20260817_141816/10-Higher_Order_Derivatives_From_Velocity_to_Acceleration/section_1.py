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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Review: The First Derivative", [
            "The derivative represents a slope.",
            "It shows the rate of change.",
            "Velocity is the first derivative."
        ])
        
        # Define function and plot
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.25 * (x - 2)**2 + 1, color=BLUE)
        axes.add(curve)
        
        # Position axes in an area that doesn't overlap the left-side text
        self.place_in_area(axes, 'B2', 'F6', scale_factor=0.3)
        self.add(axes)
        
        # Load assets
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(speedometer, 'B2', scale_factor=0.2)
        self.add(speedometer)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Animate car moving along curve
        self.place_at_grid(car, 'B2', scale_factor=0.1)
        self.play(MoveAlongPath(car, curve), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Highlight zero slope point
        zero_slope_dot = Dot(axes.c2p(2, 1), color=GREEN)
        self.play(FadeIn(zero_slope_dot))
        self.wait(2)
