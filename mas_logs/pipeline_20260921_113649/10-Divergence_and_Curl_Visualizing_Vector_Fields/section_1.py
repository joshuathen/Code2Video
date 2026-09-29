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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vector fields assign vectors to points in space.",
            "Imagine floating particles in a fluid flow.",
            "Velocity of water defines the vector field."
        ]
        self.setup_layout("Prerequisites & Intuitive Setup", lecture_lines)
        
        # Define objects
        # Use assets as requested
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        fluid_source = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg")
        fluid_sink = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg")
        
        field = ArrowVectorField(lambda pos: [0.5, 0.5, 0], x_range=[-2, 2, 0.5], y_range=[-2, 2, 0.5], colors=[WHITE])
        
        source_label = Text("Source", font_size=20, color=YELLOW)
        sink_label = Text("Sink", font_size=20, color=BLUE)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(field, 'B3', 'E6', scale_factor=0.55)
        self.play(Create(field))
        self.place_at_grid(particle, 'B2', scale_factor=0.5)
        self.play(FadeIn(particle))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        self.place_at_grid(fluid_source, 'C2', scale_factor=0.5)
        self.place_at_grid(source_label, 'A2', scale_factor=0.6)
        self.play(Create(fluid_source), Write(source_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        self.place_at_grid(fluid_sink, 'E5', scale_factor=0.5)
        self.place_at_grid(sink_label, 'F5', scale_factor=0.6)
        self.play(Create(fluid_sink), Write(sink_label))
        
        self.wait(2)
