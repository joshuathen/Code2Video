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
        lecture_lines = ["ODEs model dynamic real-world systems.", "They reveal the rules governing change.", "Physics, biology, and economics use them."]
        self.setup_layout("Summary and Real-World Bridge", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Review main points using a summary list. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg] #FFFFFF.
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        summary = VGroup(
            planet,
            Text("Model Dynamic Systems", font_size=24),
            Text("Rules of Change", font_size=24),
            Text("Multidisciplinary", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(summary, 'C2', scale_factor=0.8)
        self.play(FadeIn(summary))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Fade in a real-world application graphic. #FF0000.
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], axis_config={"include_tip": False}, x_length=3, y_length=3)
        curve = axes.plot(lambda x: 4 * np.exp(-x/1.5), color=BLUE)
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'D3', 'E5', scale_factor=0.9)
        self.play(FadeIn(graph))
        self.lecture[1].set_color("#FF0000")
        
        # === Animation for Lecture Line 3 ===
        # End by showing a clean concluding screen with a laboratory icon. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laboratory.svg] #FFFF00.
        lab = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laboratory.svg")
        conclusion = VGroup(
            lab,
            Text("Conclusion: ODEs Define Rules", font_size=32, color=YELLOW)
        ).arrange(DOWN)
        self.place_at_grid(conclusion, 'F4', scale_factor=0.7)
        self.play(Write(conclusion))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
