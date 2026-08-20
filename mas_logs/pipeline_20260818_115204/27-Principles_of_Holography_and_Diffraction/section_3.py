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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Photography records intensity, losing phase info.",
            "Holography captures both amplitude and phase.",
            "Interference patterns encode the 3D wavefront.",
            "Object and reference beams overlap precisely.",
            "The record becomes a complex hologram."
        ]
        self.setup_layout("Holography: Recording the Wavefront", lecture_lines)
        
        # Define colored line references
        colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#FF9F1C", "#A2D2FF"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        # Fix 27: Positioned img_plane at 'A4' with scale 0.6
        img_plane = Rectangle(width=2, height=1.5, color=WHITE)
        self.place_at_grid(img_plane, 'A4', scale_factor=0.6)
        self.add(img_plane)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        # Fix 28: Positioned wave at 'C4' with scale 0.6
        wave = FunctionGraph(lambda x: np.sin(x * 4), x_range=[-1, 1])
        self.place_at_grid(wave, 'C4', scale_factor=0.6)
        self.play(Create(wave))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        # Simple interference pattern sketch
        pattern = VGroup(*[Line(start=self.grid['D2'], end=self.grid['E5'], color=WHITE)])
        self.place_at_grid(pattern, 'D3')
        self.play(FadeIn(pattern))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        beam1 = Arrow(start=self.grid['A3'], end=self.grid['D3'], color=BLUE)
        beam2 = Arrow(start=self.grid['A5'], end=self.grid['D3'], color=RED)
        self.play(GrowArrow(beam1), GrowArrow(beam2))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        # Fix 29: Use place_in_area for hologram
        hologram = Square(side_length=1.5, color=PURPLE, fill_opacity=0.5)
        hologram_group = VGroup(hologram)
        self.place_in_area(hologram_group, 'E3', 'F5', scale_factor=0.7)
        self.play(FadeIn(hologram))
        self.wait(2)
