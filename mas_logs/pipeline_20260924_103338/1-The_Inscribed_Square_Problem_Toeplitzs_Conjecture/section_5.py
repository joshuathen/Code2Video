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
        lecture_lines = ["For smooth curves, it is proven.", "General cases remain an open challenge.", "Topology maps the unpredictable."]
        self.setup_layout("Conclusion & Open Frontiers", lecture_lines)
        
        # Elements
        summary_eq = MathTex(r"f(x) = 0", color=WHITE)
        icon_topology = Circle(radius=0.5, color=YELLOW, fill_opacity=0.5)
        label_topology = Text("Topology", font_size=24, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(summary_eq, 'D3', 'D4', scale_factor=1.0)
        self.play(FadeIn(summary_eq))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_in_area(icon_topology, 'B3', 'C4', scale_factor=0.9)
        self.place_at_grid(label_topology, 'C3', scale_factor=0.8)
        self.play(FadeIn(icon_topology), FadeIn(label_topology))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(
            summary_eq.animate.set_color("#00FF00"),
            icon_topology.animate.set_color("#00FF00"),
            label_topology.animate.set_color("#00FF00")
        )
        self.play(Flash(summary_eq.get_center(), color=GREEN), Flash(icon_topology.get_center(), color=GREEN))
        self.wait(2)
