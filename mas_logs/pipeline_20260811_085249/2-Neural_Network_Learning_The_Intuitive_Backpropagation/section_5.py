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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion & Takeaway", [
            "Learning finds the optimal weight combination.",
            "The network maps values, not logic.",
            "Constant correction improves overall performance."
        ])
        
        # Create a simplified neural network representation
        nodes = VGroup()
        for i in range(3):
            for j in range(3):
                node = Circle(radius=0.2, color=BLUE_B, fill_opacity=0.5)
                nodes.add(node)
        
        nodes.arrange_in_grid(rows=3, cols=3, buff=0.4)
        # Applying requested layout fix: 
        # Resolves issues 32, 34 by using B3-E6 area for balanced layout
        self.place_in_area(nodes, "B3", "E6", scale_factor=0.7)
        
        connections = VGroup()
        for i in range(0, 9, 3):
            for j in range(i, i + 2):
                line = Line(nodes[j].get_center(), nodes[j+1].get_center(), color=GRAY)
                connections.add(line)
        
        self.add(connections, nodes)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(connections), FadeIn(nodes))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(TEAL))
        self.play(nodes.animate.set_color(GREEN_B))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED_B))
        
        # Pulse nodes
        self.play(nodes.animate.set_color(WHITE), run_time=1)
        self.play(nodes.animate.set_color(BLUE_B), run_time=1)
        
        # Final text
        final_text = Text("Learning is Iterative Improvement", font_size=36, color=GOLD)
        # Applying requested layout fix: Resolves issue 33, 39
        self.place_in_area(final_text, "E2", "E5", scale_factor=0.75)
        self.play(FadeIn(final_text))
        self.wait(2)
