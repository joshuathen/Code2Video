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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Closing: The Iterative Loop", [
            "Learning is a continuous, iterative cycle.",
            "Prediction leads to loss, then weight adjustments.",
            "Repeat this loop until error disappears."
        ])

        # Animation Elements
        cycle = Circle(radius=1.2, color=BLUE)
        labels = ["Forward", "Loss", "Backprop", "Update"]
        
        # Cycle components
        loop_group = VGroup(cycle)
        for i, angle in enumerate([PI/2, 0, -PI/2, PI]):
            arrow = Arrow(start=cycle.point_at_angle(angle + PI/4), end=cycle.point_at_angle(angle), color=WHITE, buff=0)
            label = Text(labels[i], font_size=16, color=WHITE).scale(0.7)
            label.next_to(cycle.point_at_angle(angle), RIGHT if i%2==0 else LEFT, buff=0.1)
            loop_group.add(arrow, label)
        
        # Fix: Positioning in area C2-E4 (compliant with B004)
        self.place_in_area(loop_group, "C2", "E4", scale_factor=0.8)
        
        loss_val = DecimalNumber(1.0, num_decimal_places=2, color=RED)
        loss_label = Text("Current Loss:", font_size=18, color=WHITE).scale(0.7)
        loss_box = VGroup(loss_label, loss_val).arrange(RIGHT, buff=0.1)
        # Fix: Vertically aligned in grid E4 (compliant with B011/B020)
        self.place_at_grid(loss_box, "E4", scale_factor=0.7)

        # Animation sequence
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_opacity(1)
        self.lecture[0].set_color(BLUE)
        self.play(Create(loop_group))
        
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_opacity(1)
        self.lecture[1].set_color(GREEN)
        self.play(
            loss_val.animate.set_value(0.5),
            run_time=2
        )
        
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_opacity(1)
        self.lecture[2].set_color(YELLOW)
        self.play(
            loss_val.animate.set_value(0.01),
            run_time=2
        )
        self.wait(1)
