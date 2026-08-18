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
        lecture_lines = [
            "The learning loop is a four-step cycle.",
            "Forward pass, loss, backprop, then update.",
            "Iterative adjustment creates intelligent behavior."
        ]
        self.setup_layout("The Loop: Iterative Optimization", lecture_lines)
        
        # Setup circular layout objects
        labels = ["Forward", "Loss", "Backprop", "Update"]
        colors = [BLUE, RED, YELLOW, GREEN]
        nodes = VGroup()
        for i, label in enumerate(labels):
            circle = Circle(radius=0.4, color=colors[i])
            text = Text(label, font_size=16).scale(0.7) # Apply B020
            text.next_to(circle, DOWN, buff=0.1) # Apply B011
            nodes.add(VGroup(circle, text))
        
        # Brain icon (Asset)
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        brain.set_color(WHITE)
        
        # Circular Layout Setup
        nodes.arrange_in_grid(buff=0.1) # Replaced invalid config dictionary
        brain.move_to(nodes.get_center())
        
        # Use place_in_area as recommended by Critic (B004: use col 4-6)
        # Using C3-E5 was suggested, but B004 mandates col 4-6 for primary animation.
        # Adjusted to be safe: columns 3-6.
        loop_group = VGroup(nodes, brain)
        self.place_in_area(loop_group, 'C3', 'E5', scale_factor=0.9)
        
        # Arrows (B035: directional)
        arrows = VGroup()
        for i in range(4):
            start = nodes[i].get_center()
            end = nodes[(i + 1) % 4].get_center()
            arrow = Arrow(start, end, buff=0.3, color=WHITE, stroke_width=3)
            arrows.add(arrow)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(nodes), FadeIn(brain))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(Create(arrows), run_time=1.5)
        self.play(Rotating(arrows, radians=2*PI, about_point=nodes.get_center(), run_time=2))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Use brain icon color change for B035/Story
        self.play(brain.animate.set_color("#32CD32"), run_time=1)
        self.lecture[2].set_color(GREEN)
