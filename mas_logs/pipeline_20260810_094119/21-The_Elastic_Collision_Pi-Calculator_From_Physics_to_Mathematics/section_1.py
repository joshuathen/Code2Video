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
        self.setup_layout("Introduction: The Paradox of Particles", [
            "Two blocks collide on a frictionless surface.", 
            "Mass A is one, Mass B is larger.", 
            "How many collisions until they stop moving?", 
            "Like a rabbit hitting a massive elephant.", 
            "Let's count the bounces."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Two blocks collide on a frictionless surface.
        block_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg", color="#4DA6FF")
        block_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/elephant.svg", color="#4DA6FF")
        
        blocks = VGroup(block_a, block_b).arrange(RIGHT, buff=2)
        self.place_in_area(blocks, "B4", "D6", scale_factor=0.5)
        self.play(FadeIn(blocks))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Mass A is one, Mass B is larger.
        label_a = Tex("M", color="#4DA6FF").next_to(block_a, UP)
        label_b = Tex(r"$100^N M$", color="#4DA6FF").next_to(block_b, UP)
        self.play(Write(label_a), Write(label_b))
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        # How many collisions until they stop moving?
        # Animate block A moving toward block B
        self.play(block_a.animate.shift(RIGHT * 1.5))
        self.play(FadeOut(block_a), FadeOut(block_b), FadeOut(label_a), FadeOut(label_b))
        
        # Reset and show impact
        self.place_in_area(blocks, "B4", "D6", scale_factor=0.5)
        self.play(FadeIn(blocks))
        impact_text = Text("Collision 1", font_size=24, color=YELLOW)
        self.place_at_grid(impact_text, "C3")
        self.play(Write(impact_text))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Like a rabbit hitting a massive elephant.
        rabbit_text = Text("Rabbit", color=BLUE, font_size=20)
        elephant_text = Text("Elephant", color=RED, font_size=20)
        self.place_at_grid(rabbit_text, "C4", scale_factor=0.8)
        self.place_at_grid(elephant_text, "C6", scale_factor=0.8)
        self.play(FadeIn(rabbit_text), FadeIn(elephant_text))
        self.lecture[3].set_color(GREEN)

        # === Animation for Lecture Line 5 ===
        # Let's count the bounces.
        bounce_count = Text("Bounces: 0", font_size=30)
        self.place_at_grid(bounce_count, "E5", scale_factor=0.9)
        self.play(Write(bounce_count))
        self.lecture[4].set_color(PURPLE)
        self.wait(1)
