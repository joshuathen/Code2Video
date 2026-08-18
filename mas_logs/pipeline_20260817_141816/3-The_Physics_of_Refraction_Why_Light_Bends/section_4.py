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
        self.setup_layout("Real-World Application: The Archer Fish", [
            "Archer fish navigate refraction to catch prey.",
            "Light bends at the water's surface.",
            "The fish compensates for the apparent position."
        ])
        
        # Asset paths
        fish_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg"
        prey_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/prey.svg"
        
        fish = SVGMobject(fish_asset, color=WHITE)
        prey = SVGMobject(prey_asset)
        
        water_surface = Line(self.grid["C1"], self.grid["C6"], color=BLUE)
        water = Rectangle(width=5.0, height=3.0, fill_color=BLUE, fill_opacity=0.3, stroke_width=0)
        water.next_to(water_surface, DOWN, buff=0)
        
        self.place_at_grid(fish, "E3", scale_factor=0.3)
        self.place_at_grid(prey, "A4", scale_factor=0.3)
        
        # Refraction lines
        ray_to_surface = Line(prey.get_center(), self.grid["C4"], color=WHITE)
        ray_to_fish = Line(self.grid["C4"], fish.get_center(), color=WHITE)
        refraction_path = VGroup(ray_to_surface, ray_to_fish)
        
        apparent_prey = Dot(self.grid["A4"], color="#00FFFF", radius=0.1) # Position A4 as per suggestion 36
        virtual_ray = DashedLine(apparent_prey.get_center(), self.grid["C4"], color="#00FFFF")
        
        # Animation groups for critic fixes
        fish_refraction_animation = VGroup(refraction_path)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.add(water, water_surface)
        self.play(FadeIn(fish), FadeIn(prey))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.place_in_area(fish_refraction_animation, 'C1', 'F3', scale_factor=0.6)
        self.play(Create(refraction_path))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.place_at_grid(apparent_prey, 'A4', scale_factor=0.7)
        self.play(FadeIn(apparent_prey), Create(virtual_ray))
        
        self.wait(2)
