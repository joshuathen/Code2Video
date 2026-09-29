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
        self.setup_layout("Architectural Workflow: Encoder & Decoder", [
            "The architecture has two main parts.",
            "Encoders digest and encode the input.",
            "Decoders generate the output step-by-step."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Visualizing two main parts
        part1 = Rectangle(width=2, height=1, color=BLUE).set_fill(BLUE, opacity=0.3)
        label1 = Text("Encoder", font_size=24)
        part2 = Rectangle(width=2, height=1, color=GREEN).set_fill(GREEN, opacity=0.3)
        label2 = Text("Decoder", font_size=24)
        
        self.place_at_grid(part1, 'B3', scale_factor=0.8)
        self.place_at_grid(label1, 'B1', scale_factor=0.7)
        self.place_at_grid(part2, 'E3', scale_factor=0.8)
        self.place_at_grid(label2, 'E1', scale_factor=0.7)
        
        architecture = VGroup(part1, label1, part2, label2)
        
        self.play(FadeIn(architecture))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Representing Encoder digesting input
        arrow = Arrow(start=self.grid['B3'], end=self.grid['B5'], color=WHITE)
        input_data = Text("Input", font_size=20)
        self.place_at_grid(input_data, 'B6')
        
        self.play(Create(arrow), FadeIn(input_data))
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Representing Decoder generating output
        arrow2 = Arrow(start=self.grid['E3'], end=self.grid['E5'], color=WHITE)
        output_data = Text("Output", font_size=20)
        self.place_at_grid(output_data, 'E6')
        
        self.play(Create(arrow2), FadeIn(output_data))
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(1)
