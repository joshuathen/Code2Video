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
        self.setup_layout("The MLP as a Key-Value Memory", [
            "MLPs function as key-value memory.",
            "First layers detect specific input patterns.",
            "Second layers project associated factual values."
        ])

        # Assets
        microchip = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg", color=WHITE)
        memory_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/memory.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        input_layer = VGroup(*[Circle(radius=0.15, color=BLUE) for _ in range(3)]).arrange(DOWN, buff=0.5)
        hidden_layer = VGroup(*[Circle(radius=0.15, color=GREEN) for _ in range(3)]).arrange(DOWN, buff=0.5)
        connections = VGroup()
        for i in input_layer:
            for h in hidden_layer:
                connections.add(Line(i.get_right(), h.get_left(), stroke_width=1, color=GRAY))
        
        mlp_group = VGroup(connections, input_layer, hidden_layer, microchip)
        microchip.next_to(mlp_group, UP)
        
        self.place_in_area(mlp_group, "B4", "D6", scale_factor=0.65)
        
        self.play(Create(mlp_group))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(
            connections[0].animate.set_color("#FF4500").set_stroke(width=3),
            connections[4].animate.set_color("#FF4500").set_stroke(width=3)
        )
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        memory_label = Text("Memory", font_size=24, color=WHITE)
        
        # Group with asset
        memory_group = VGroup(memory_label, memory_icon).arrange(RIGHT, buff=0.2)
        self.place_at_grid(memory_group, "E5", scale_factor=0.7)
        
        self.play(FadeIn(memory_group))
        self.play(
            input_layer[0].animate.set_fill(WHITE, opacity=1),
            hidden_layer[1].animate.set_fill(WHITE, opacity=1)
        )
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
