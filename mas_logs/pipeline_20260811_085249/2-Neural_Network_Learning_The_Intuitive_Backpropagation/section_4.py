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
        self.setup_layout("The Learning Cycle: Iteration", [
            "Learning is an iterative cycle.", 
            "Predict, calculate error, then update weights.", 
            "Repeated updates create an intelligent model."
        ])

        # Define cycle elements
        circle = Circle(radius=1.5, color=BLUE)
        labels = [Text("Predict", font_size=24), Text("Calculate Error", font_size=24), Text("Update", font_size=24)]
        
        # Load asset
        weight_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weights.svg")
        weight_icon.set_color(WHITE)
        
        # Position around circle
        self.place_at_grid(circle, 'C4')
        labels[0].move_to(self.grid['B4'])
        labels[1].move_to(self.grid['D6'])
        labels[2].move_to(self.grid['D2'])
        
        self.place_at_grid(weight_icon, 'D2', scale_factor=0.5)
        
        learning_cycle_group = VGroup(circle, *labels, weight_icon)
        self.place_in_area(learning_cycle_group, 'A4', 'F6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(learning_cycle_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW), self.lecture[0].animate.set_color(WHITE))
        # Highlight "Update" icon with pulse
        self.play(weight_icon.animate.set_color("#FFD700"), run_time=0.5)
        self.play(Indicate(weight_icon, color=GOLD, scale_factor=1.2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW), self.lecture[1].animate.set_color(WHITE))
        # Simulate convergence/repetition
        for _ in range(3):
            self.play(Rotate(learning_cycle_group, angle=2*PI/3, run_time=0.8))
        self.wait(1)
