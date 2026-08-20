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
        self.setup_layout("The MLP Mechanism: Key-Value Memories", [
            "Input vector activates a specific key.",
            "Neurons act like locks for specific patterns.",
            "Matching keys trigger value outputs."
        ])
        self.lecture.set_opacity(0)
        
        # Assets
        lock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg")
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg")

        # Create MLP representation
        layer1 = VGroup(*[Rectangle(width=0.5, height=0.5, color="#ADD8E6", fill_opacity=0.5) for _ in range(3)])
        layer1.arrange(DOWN, buff=0.1)
        self.place_at_grid(layer1, 'B2', scale_factor=0.7)
        
        # Label 1
        label1 = Text("Key", font_size=24, color=WHITE)
        self.place_at_grid(label1, 'B1', scale_factor=0.6)
        label1.next_to(layer1, LEFT, buff=0.1)
        
        layer2 = VGroup(*[Rectangle(width=0.5, height=0.5, color="#ADD8E6", fill_opacity=0.5) for _ in range(3)])
        layer2.arrange(DOWN, buff=0.1)
        self.place_at_grid(layer2, 'E2', scale_factor=0.7)

        # Label 2
        label2 = Text("Value", font_size=24, color=WHITE)
        self.place_at_grid(label2, 'E1', scale_factor=0.6)
        label2.next_to(layer2, LEFT, buff=0.1)

        # Highlight nodes
        nodes1 = VGroup(*[Dot(color="#00FF00") for _ in range(3)])
        for i, dot in enumerate(nodes1):
            dot.move_to(layer1[i].get_center())
            
        nodes2 = VGroup(*[Dot(color="#00FF00") for _ in range(3)])
        for i, dot in enumerate(nodes2):
            dot.move_to(layer2[i].get_center())

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), Write(self.lecture[0]))
        self.lecture[0].set_color("#ADD8E6")
        lock.scale(0.3).next_to(layer1, UP)
        self.play(Create(layer1), FadeIn(lock))
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), FadeIn(nodes1), FadeIn(nodes2))
        self.lecture[1].set_color("#00FF00")
        
        # === Animation for Lecture Line 3 ===
        arrow = Arrow(layer1.get_right(), layer2.get_left(), color=WHITE)
        self.place_in_area(arrow, 'B3', 'E3', scale_factor=0.8)
        key_icon.scale(0.3).next_to(arrow, UP)
        self.play(FadeIn(self.lecture[2]), Create(arrow), FadeIn(key_icon))
        self.lecture[2].set_color("#FFFFFF")
        
        self.wait(2)
