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
        self.setup_layout("The Analogy: The Guessing Game", [
            "Neural networks start as simple guessing machines.",
            "They classify inputs into random, wrong categories.",
            "A robot guesses cat versus dog pixels."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Neural network nodes initialized to white (random state)
        nodes = VGroup(*[Circle(radius=0.2, color=WHITE, fill_opacity=1) for _ in range(3)])
        nodes.arrange(DOWN, buff=0.5)
        self.place_at_grid(nodes, 'C4')
        self.play(FadeIn(nodes))

        # === Animation for Lecture Line 2 ===
        # Cat and Dog icons
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        dog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg")
        
        self.place_at_grid(cat, 'C5', scale_factor=0.3)
        self.place_at_grid(dog, 'D5', scale_factor=0.3)
        
        cat_label = Text("Cat", font_size=18, color=RED).scale(0.7).next_to(cat, RIGHT, buff=0.1)
        dog_label = Text("Dog", font_size=18, color=RED).scale(0.7).next_to(dog, RIGHT, buff=0.1)
        
        self.play(FadeIn(cat), FadeIn(dog), Write(cat_label), Write(dog_label))
        self.play(cat_label.animate.set_color(RED), dog_label.animate.set_color(RED))

        # === Animation for Lecture Line 3 ===
        # Robot pixel input (2x2)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'E4', scale_factor=0.4)
        
        pixels = VGroup(*[Square(side_length=0.3, color=WHITE, fill_opacity=0.3) for _ in range(4)])
        pixels.arrange_in_grid(2, 2, buff=0.05)
        self.place_at_grid(pixels, 'E5')
        
        self.play(FadeIn(robot), FadeIn(pixels))
        
        # Fluctuating highlight
        pixel_highlight = SurroundingRectangle(pixels, color=YELLOW, buff=0.05)
        self.play(Create(pixel_highlight))
        self.play(Flash(pixels, color=YELLOW, line_length=0.1, num_lines=8))
        self.wait(2)
