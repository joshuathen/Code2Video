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
        lecture_lines = [
            "Backpropagation is like assigning blame for mistakes.",
            "We start at the output error point.",
            "The chain rule moves blame backward through layers.",
            "Each weight receives its contribution to the error.",
            "This blame helps us adjust individual weights precisely."
        ]
        self.setup_layout("The Heart of the Lesson: Backpropagation Intuition", lecture_lines)
        
        # Nodes
        nodes = VGroup(*[Circle(radius=0.2, color=WHITE) for _ in range(9)])
        # Input layer (B3, C3, D3), Hidden (B4, C4, D4), Output (B5, C5, D5) - Adjusted based on critic
        self.place_at_grid(nodes[0], 'B3')
        self.place_at_grid(nodes[1], 'C3')
        self.place_at_grid(nodes[2], 'D3')
        
        self.place_at_grid(nodes[3], 'B4')
        self.place_at_grid(nodes[4], 'C4')
        self.place_at_grid(nodes[5], 'D4')

        self.place_at_grid(nodes[6], 'B5')
        self.place_at_grid(nodes[7], 'C5')
        self.place_at_grid(nodes[8], 'D5')
        
        network_group = VGroup(nodes)
        self.place_in_area(network_group, 'B3', 'E5', scale_factor=0.9)

        # Edges
        lines = VGroup()
        for i in [0,1,2]:
            for j in [3,4,5]:
                lines.add(Line(nodes[i].get_right(), nodes[j].get_left(), color=GRAY))
        for i in [3,4,5]:
            for j in [6,7,8]:
                lines.add(Line(nodes[i].get_right(), nodes[j].get_left(), color=GRAY))
        
        self.add(lines, nodes)

        # Asset loads
        heart_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heart.svg")
        heart_asset.set_height(0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        output_node_visual = heart_asset.copy().set_color("#FF0000")
        output_node_visual.move_to(nodes[8].get_center())
        self.add(output_node_visual)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF9900"))
        error_label = Text("Error!", font_size=20, color="#FF9900")
        self.place_at_grid(error_label, 'D6', scale_factor=1.0).shift(0.5 * DOWN)
        self.add(error_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Chain rule animation representation
        path = VGroup(lines[6], lines[7], lines[8], nodes[5], nodes[4], nodes[3])
        self.play(path.animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFCC00"))
        # Weight adjustments
        self.play(lines.animate.set_color("#FFCC00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.play(nodes.animate.set_color("#00FF00"))
        final_heart = heart_asset.copy().set_color("#FFFFFF")
        final_heart.move_to(nodes[8].get_center())
        self.play(FadeIn(final_heart))
        self.wait(1)
