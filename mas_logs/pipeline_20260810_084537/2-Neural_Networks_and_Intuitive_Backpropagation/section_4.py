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
        self.setup_layout("Backpropagation: The Credit Assignment Problem", 
                          ["Backpropagation sends the error back through layers.", 
                           "We find which weight caused the error.", 
                           "Like tracing a chain of falling dominoes."])
        
        # Mobjects
        network = VGroup(*[Circle(radius=0.15, color=BLUE) for _ in range(6)])
        network.arrange_in_grid(2, 3, buff=0.4)
        
        weights = VGroup(*[Line(network[i].get_center(), network[j].get_center(), color=WHITE) 
                          for i in range(3) for j in range(3, 6)])
        
        network_group = VGroup(network, weights)
        
        dominoes = VGroup(*[Rectangle(height=0.3, width=0.08, color=GREEN) for _ in range(4)])
        dominoes.arrange(RIGHT, buff=0.2)
        
        # Combined group for coherent narrative flow
        combined_group = VGroup(network_group, dominoes).arrange(DOWN, buff=0.8)
        
        # Layout
        self.place_in_area(combined_group, 'B2', 'E5', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(network), Create(weights))
        self.play(self.lecture[0].animate.set_color("#FF6666"))

        # === Animation for Lecture Line 2 ===
        error_signal = Arrow(network[5].get_center(), network[2].get_center(), color=RED)
        self.play(Create(error_signal))
        # Highlight a weight
        self.play(weights[0].animate.set_color(YELLOW).scale(1.1))
        self.play(self.lecture[1].animate.set_color("#FF6666"))

        # === Animation for Lecture Line 3 ===
        self.play(Create(dominoes))
        self.play(self.lecture[2].animate.set_color("#66FF66"))
        self.wait(2)
