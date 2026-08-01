from manim import *
import numpy as np

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
        # Define lecture content based on Stage-3 requirements
        lines = [
            'Archerfish must master physics to catch their prey.',
            "Light bending at the surface shifts the fly's image.",
            'The fish aims below the image to hit reality.'
        ]
        
        self.setup_layout("The Archerfish: Refraction in the Wild", lines)

        # Pre-define key positions and colors
        fish_color = "#FFA500"
        fly_color = "#A9A9A9"
        ghost_color = "#D3D3D3"
        water_color = "#1E90FF"
        ray_color = YELLOW
        aim_color = RED
        
        # Grid points
        fly_pos = self.grid['B2']
        int_pos = self.grid['D4'] # Intersection point on the surface
        fish_pos = self.grid['F4']
        ghost_pos = self.grid['B4']
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(fish_color)
        
        # Create Water Area
        water_bg = Rectangle(width=6.5, height=3.5, fill_opacity=0.3, fill_color=water_color, stroke_width=0)
        self.place_in_area(water_bg, 'D1', 'F6')
        
        # Replacement for missing SVG Fish
        fish = Triangle(fill_opacity=1).rotate(-90 * DEGREES)
        fish.set_color(fish_color)
        self.place_at_grid(fish, 'F4', scale_factor=0.4)
        fish_label = Text("Archerfish", font_size=18, color=fish_color)
        self.place_at_grid(fish_label, 'F5', scale_factor=0.8)
        
        # Replacement for missing SVG Fly
        fly = Circle(radius=0.15, fill_opacity=1)
        fly.set_color(fly_color)
        self.place_at_grid(fly, 'B2', scale_factor=1.0)
        fly_label = Text("Real Fly", font_size=18, color=fly_color)
        self.place_at_grid(fly_label, 'A2', scale_factor=0.8)
        
        self.play(
            FadeIn(water_bg), 
            DrawBorderThenFill(fish), 
            FadeIn(fish_label), 
            GrowFromCenter(fly), 
            FadeIn(fly_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ghost_color)
        
        # Replacement for missing SVG Ghost Fly
        ghost_fly = Circle(radius=0.15, fill_opacity=1)
        ghost_fly.set_color(ghost_color).set_opacity(0.6)
        self.place_at_grid(ghost_fly, 'B4', scale_factor=1.0)
        ghost_label = Text("Apparent Fly", font_size=18, color=ghost_color)
        self.place_at_grid(ghost_label, 'A4', scale_factor=0.8)
        
        # Normal line at the interface
        normal = Line(self.grid['C4'], self.grid['E4'], color=WHITE, stroke_width=2).set_opacity(0.5)
        
        # Trace Light Ray: Fly -> Surface -> Fish Eye
        ray_air = Line(fly_pos, int_pos, color=ray_color, stroke_width=4)
        ray_water = Line(int_pos, fish_pos, color=ray_color, stroke_width=4)
        
        # Dashed line showing apparent path: Fish -> Surface -> Ghost Fly
        apparent_path = DashedLine(fish_pos, ghost_pos, color=ghost_color, dash_length=0.1)
        
        self.play(Create(normal))
        self.play(Create(ray_air), Create(ray_water))
        self.play(Create(apparent_path), FadeIn(ghost_fly), FadeIn(ghost_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(aim_color)
        
        # Show the fish's aim (straight line to the real fly)
        aim_arrow = Arrow(start=fish_pos, end=fly_pos, color=aim_color, buff=0.3, stroke_width=6)
        compensated_aim_label = Text("Compensated Aim", font_size=16, color=aim_color)
        self.place_at_grid(compensated_aim_label, 'E2', scale_factor=0.7)
        
        # Replacement for missing Image Droplet
        droplet = Dot(color=BLUE, radius=0.1)
        self.place_at_grid(droplet, 'F4', scale_factor=1.0)
        
        self.play(GrowArrow(aim_arrow), Write(compensated_aim_label))
        self.wait(0.5)
        self.play(FadeIn(droplet))
        self.play(droplet.animate.move_to(fly_pos), run_time=1.5, rate_func=linear)
        self.play(FadeOut(droplet), Flash(fly_pos, color=aim_color))
        self.wait(2)
