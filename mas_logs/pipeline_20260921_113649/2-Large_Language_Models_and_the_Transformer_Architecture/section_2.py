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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Computers only understand numbers, not text.",
            "Words are mapped to high-dimensional coordinates.",
            "Similar concepts are placed close together."
        ]
        self.setup_layout("Prerequisite: Embeddings", lecture_lines)
        
        # Define visual elements
        dot_dog = Dot(color=BLUE)
        label_dog = Text("Dog", font_size=20)
        dot_puppy = Dot(color=BLUE)
        label_puppy = Text("Puppy", font_size=20)
        dot_spaceship = Dot(color=RED)
        label_spaceship = Text("Spaceship", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        binary_code = Text("01010101", font_size=30, color=GRAY)
        self.place_at_grid(binary_code, 'B3', scale_factor=0.9)
        self.play(FadeIn(binary_code))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeOut(binary_code))
        
        # Setup grid/axes for vector space
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False}).scale(0.7)
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.8)
        self.play(Create(axes))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Place words near each other
        pos_dog = axes.c2p(-0.5, -0.5)
        pos_puppy = axes.c2p(-0.3, -0.3)
        pos_spaceship = axes.c2p(1.5, 1.5)
        
        dot_dog.move_to(pos_dog)
        dot_puppy.move_to(pos_puppy)
        dot_spaceship.move_to(pos_spaceship)
        
        label_dog.next_to(dot_dog, UP, buff=0.1)
        label_puppy.next_to(dot_puppy, UP, buff=0.1)
        label_spaceship.next_to(dot_spaceship, UP, buff=0.1)
        
        # Combine labels for unified group processing as requested by Critic
        labels = VGroup(label_dog, label_puppy, label_spaceship)
        self.play(FadeIn(dot_dog, dot_puppy, dot_spaceship, labels))
        self.wait(2)
