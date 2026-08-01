from manim import *
import random

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
        # Paths to assets
        lion_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg"
        penguin_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/penguin.svg"
        elephant_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/elephant.svg"

        self.setup_layout(
            "The Big Scramble: Identifying the Problem", 
            [
                "Look at all these jumping animals!", 
                "It is hard to count them while they move.", 
                "How many elephants do you see?"
            ]
        )

        # Create 15 animal icons
        animals = []
        elephants = []
        
        # 5 Lions
        for _ in range(5):
            animals.append(SVGMobject(lion_path, color=ORANGE))
        # 5 Penguins
        for _ in range(5):
            animals.append(SVGMobject(penguin_path, color=BLUE_B))
        # 5 Elephants
        for _ in range(5):
            e = SVGMobject(elephant_path, color=GRAY)
            animals.append(e)
            elephants.append(e)

        # All grid positions
        grid_keys = [f"{r}{c}" for r in "ABCDEF" for c in "123456"]
        random.seed(42)  # For reproducibility
        initial_positions = random.sample(grid_keys, 15)

        # Place animals
        for animal, pos in zip(animals, initial_positions):
            self.place_at_grid(animal, pos, scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Define rhythmic jump animation
        def jump_update(mobject, dt):
            mobject.shift(UP * 0.1 * np.sin(self.time * 10))

        for animal in animals:
            animal.add_updater(jump_update)
        
        self.add(*animals)
        self.play(FadeIn(VGroup(*animals)), run_time=1)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Move animals to new random positions while jumping
        new_positions = random.sample(grid_keys, 15)
        move_animations = []
        for animal, pos_key in zip(animals, new_positions):
            move_animations.append(animal.animate.move_to(self.grid[pos_key]))
        
        self.play(*move_animations, run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Pulse red circle around two elephants sequentially
        for i in range(2):
            elephant = elephants[i]
            circle = Circle(radius=0.4, color="#FF0000", stroke_width=4)
            circle.move_to(elephant.get_center())
            
            # Anchor circle to elephant as it jumps
            circle.add_updater(lambda c, e=elephant: c.move_to(e.get_center()))
            
            self.play(Create(circle), run_time=0.5)
            self.play(circle.animate.scale(1.2), run_time=0.3, rate_func=there_and_back)
            self.play(FadeOut(circle), run_time=0.5)
        
        self.wait(2)

        # Cleanup updaters to prevent errors at end
        for animal in animals:
            animal.remove_updater(jump_update)
