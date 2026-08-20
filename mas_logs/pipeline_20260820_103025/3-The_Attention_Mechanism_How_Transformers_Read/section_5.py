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
        self.setup_layout("Conclusion: From Attention to Intelligence", [
            "Stacking layers captures complex deep relationships.",
            "This powers modern Large Language Models.",
            "Attention transforms text into intelligent insight."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        # Asset: server.svg
        server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color=WHITE)
        layers = VGroup(*[Rectangle(height=0.5, width=2.0, color="#FF00FF") for _ in range(3)])
        layers.arrange(UP, buff=0.1)
        self.place_at_grid(layers, 'E5')
        self.place_at_grid(server, 'F5', scale_factor=0.5)
        
        self.play(FadeIn(server), Create(layers), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#00FFFF")
        arrows = VGroup(*[DoubleArrow(start=layers[i].get_center(), end=layers[i+1].get_center(), color="#00FFFF", buff=0.1) for i in range(2)])
        self.play(Create(arrows), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FFFF00")
        
        # Morph into intelligence icon and transition to server.svg
        icon = Star(color="#FFFF00", fill_opacity=1)
        # Fix: Animation crowding per VideoCritic
        self.place_in_area(icon, 'B3', 'E5', scale_factor=1.2)
        
        self.play(ReplacementTransform(VGroup(layers, arrows), icon))
        self.play(icon.animate.scale(0.5).move_to(self.grid['D4']), run_time=1)
        self.play(ReplacementTransform(icon, server), run_time=1)
        self.wait(1)
