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
        self.setup_layout("The 'Credit Assignment' Visualization", [
            "Gradients flow backward through the network.",
            "Thicker arrows signify stronger error influence.",
            "We map responsibility to every connection.",
            "This visualization highlights critical neural pathways.",
            "Precision in credit assignment drives better learning."
        ])
        
        # Setup visual elements: A simplified neural network path (Nodes and connections)
        network = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg")
        self.place_in_area(network, 'B4', 'E6', scale_factor=0.8)
        
        nodes = VGroup(*[Circle(radius=0.15, color=BLUE, fill_opacity=0.5) for _ in range(6)])
        self.place_at_grid(nodes[0], 'B4')
        self.place_at_grid(nodes[1], 'B6')
        self.place_at_grid(nodes[2], 'D4')
        self.place_at_grid(nodes[3], 'D6')
        # Applying fix from Issue 31
        self.place_at_grid(nodes[4], 'E4', scale_factor=0.9)
        self.place_at_grid(nodes[5], 'E6', scale_factor=0.9)
        
        # Creating lines
        lines = VGroup(
            Line(nodes[0].get_center(), nodes[2].get_center(), stroke_width=4),
            Line(nodes[1].get_center(), nodes[3].get_center(), stroke_width=4),
            Line(nodes[2].get_center(), nodes[4].get_center(), stroke_width=4),
            Line(nodes[3].get_center(), nodes[5].get_center(), stroke_width=4)
        )
        
        self.add(network, nodes, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Arrows flowing backward
        arrows = VGroup(*[Arrow(lines[i].get_end(), lines[i].get_start(), color=YELLOW, buff=0.1) for i in range(len(lines))])
        self.play(Create(arrows))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Adjust line thickness to show influence
        self.play(
            lines[0].animate.set_stroke(width=8),
            lines[1].animate.set_stroke(width=2),
            lines[2].animate.set_stroke(width=10),
            lines[3].animate.set_stroke(width=3)
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Labels with Fix from Issue 29
        labels = VGroup(*[MathTex(f"w_{{{i+1}}}", font_size=20) for i in range(len(lines))])
        self.place_at_grid(labels[0], 'B3', scale_factor=0.7)
        self.place_at_grid(labels[1], 'C3', scale_factor=0.7)
        self.place_at_grid(labels[2], 'D3', scale_factor=0.7)
        self.place_at_grid(labels[3], 'E4', scale_factor=0.7)
        
        self.play(Write(labels))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight critical path with Fix from Issue 30
        path_highlight = SurroundingRectangle(VGroup(lines[2], nodes[2], nodes[4]), color="#FF4500", buff=0.1)
        self.place_in_area(path_highlight, 'C2', 'D4', scale_factor=0.9)
        self.play(Create(path_highlight))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(FadeOut(path_highlight), FadeOut(arrows), FadeOut(labels), FadeOut(nodes), FadeOut(lines), FadeOut(network))
        self.wait(1)
