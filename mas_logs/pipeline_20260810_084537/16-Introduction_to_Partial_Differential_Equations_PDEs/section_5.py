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
        self.setup_layout("Summary and Real-World Impact", [
            "PDEs model complex, multi-dimensional systems.",
            "Applications: weather, fluids, and finance.",
            "Nature unfolds through these dynamic equations."
        ])
        
        # === Animation for Lecture Line 1 ===
        # PDEs model complex, multi-dimensional systems.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Visual: Text \"PDEs govern complex physical systems\"
        complex_text = Text("PDEs govern complex physical systems", font_size=32, color=WHITE)
        self.place_in_area(complex_text, 'C2', 'C5', scale_factor=0.75)
        self.play(Write(complex_text))
        
        # === Animation for Lecture Line 2 ===
        # Applications: weather, fluids, and finance.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Visual: Flash key terms 'Heat', 'Fluid', 'Wave' with icons
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        faucet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg")
        ocean = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ocean.svg")
        
        term1 = VGroup(Text("Heat", color="#00FFFF"), thermometer).arrange(UP)
        term2 = VGroup(Text("Fluid", color="#00FFFF"), faucet).arrange(UP)
        term3 = VGroup(Text("Wave", color="#00FFFF"), ocean).arrange(UP)
        
        highlight_group = VGroup(term1, term2, term3).arrange(RIGHT, buff=0.8)
        
        self.place_in_area(highlight_group, 'B2', 'B5', scale_factor=0.65)
        self.play(FadeIn(highlight_group))

        # === Animation for Lecture Line 3 ===
        # Nature unfolds through these dynamic equations.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Visual: Montage of system imagery
        montage = VGroup(
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ocean.svg")
        ).arrange(RIGHT, buff=0.5)
        
        self.place_in_area(montage, 'D3', 'D4', scale_factor=0.55)
        self.play(Create(montage))
        
        self.wait(2)
