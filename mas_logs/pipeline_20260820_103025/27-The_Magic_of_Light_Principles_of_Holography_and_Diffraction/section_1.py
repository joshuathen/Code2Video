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
        self.setup_layout("Prerequisite: The Wave Nature of Light", [
            "Light waves exhibit natural superposition.", 
            "Coherent lasers ensure stable interference.", 
            "Visualizing overlapping light crests."
        ])
        self.lecture.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg] for point source
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#FFFFFF")
        
        source1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg").set_color(WHITE)
        source2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg").set_color(WHITE)
        
        self.place_at_grid(source1, 'A1', scale_factor=0.3)
        self.place_at_grid(source2, 'A6', scale_factor=0.3)
        
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), color="#00FFFF")
        self.place_in_area(wave1, 'B1', 'B6', scale_factor=0.9)
        
        self.play(FadeIn(source1), FadeIn(source2), Create(wave1))
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color("#FF00FF")
        
        # Laser focus effect using asset
        laser_effect = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg").set_color("#FF00FF")
        self.place_at_grid(laser_effect, 'C3', scale_factor=0.5)
        self.play(GrowFromCenter(laser_effect))
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color("#FFFFFF")
        
        # Overlapping sine waves
        wave_a = FunctionGraph(lambda x: 0.3 * np.sin(4 * x), color="#FFFFFF")
        wave_b = FunctionGraph(lambda x: 0.3 * np.sin(4 * x + PI/2), color="#FFFFFF")
        
        self.place_in_area(wave_a, 'D1', 'E6', scale_factor=0.85)
        self.place_in_area(wave_b, 'F1', 'F6', scale_factor=0.85)
        
        self.play(Create(wave_a), Create(wave_b))
        self.wait(2)
