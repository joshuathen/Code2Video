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
        self.setup_layout("Conclusion and Intuition", [
            "Phase space rotations yield discrete counts.",
            "Pi emerges from conservation laws.",
            "Geometry and physics are deeply linked."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in Phase Space diagram showing circular trajectories. Color: #FFFFFF.
        phase_diagram = Circle(radius=1.0, color=WHITE)
        self.place_at_grid(phase_diagram, 'C3', scale_factor=0.6)
        label = Text("Phase Space", color=WHITE, font_size=20)
        label.next_to(phase_diagram, UP)
        self.play(FadeIn(phase_diagram), FadeIn(label))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Flash the intersection of geometry and physics nodes. Color: #00FFFF.
        node1 = Dot(color="#00FFFF").move_to(phase_diagram.get_center() + LEFT * 0.5)
        node2 = Dot(color="#00FFFF").move_to(phase_diagram.get_center() + RIGHT * 0.5)
        node3 = Dot(color="#00FFFF").move_to(phase_diagram.get_center() + UP * 0.5)
        
        node_label1 = Text("Physics", color="#00FFFF", font_size=16).next_to(node1, LEFT)
        node_label2 = Text("Geometry", color="#00FFFF", font_size=16).next_to(node2, RIGHT)
        
        self.play(FadeIn(node1), FadeIn(node2), FadeIn(node3), FadeIn(node_label1), FadeIn(node_label2))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        # Fade out all elements to black background.
        self.play(FadeOut(phase_diagram), FadeOut(label), FadeOut(node1), FadeOut(node2), FadeOut(node3), FadeOut(node_label1), FadeOut(node_label2))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(1)
