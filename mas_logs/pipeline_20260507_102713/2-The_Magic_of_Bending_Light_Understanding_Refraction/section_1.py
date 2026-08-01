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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup with provided script
        title = "The Visual Illusion Hook"
        lines = [
            'First, we have a simple glass of water.',
            'When we dip a pencil into the water...',
            '...it looks broken at the surface because of light.',
            'Now, look at this coin at the bottom.',
            'Light bending makes the coin seem closer than usual.'
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.lecture[0].set_color("#ADD8E6")
        
        glass_width = 3.0
        glass_height = 3.5
        water_height = 2.0
        
        # Create glass outline (#FFFFFF) and water (#ADD8E6)
        glass_rect = VGroup(
            Line([-glass_width/2, glass_height/2, 0], [-glass_width/2, -glass_height/2, 0]),
            Line([-glass_width/2, -glass_height/2, 0], [glass_width/2, -glass_height/2, 0]),
            Line([glass_width/2, -glass_height/2, 0], [glass_width/2, glass_height/2, 0])
        ).set_color("#FFFFFF")
        
        water = Rectangle(
            width=glass_width, 
            height=water_height, 
            fill_color="#ADD8E6", 
            fill_opacity=0.5, 
            stroke_width=0
        )
        water.move_to(glass_rect[1].get_center() + UP * (water_height / 2))
        
        container = VGroup(water, glass_rect)
        # Fix Issue 29: Position container in C2-F5 area
        self.place_in_area(container, 'C2', 'F5', scale_factor=0.8)
        
        self.play(Create(glass_rect))
        self.play(FadeIn(water))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00") # Yellow for pencil
        
        pencil_angle = -45 * DEGREES
        pencil_length = 3.8
        # Create pencil in two parts to allow for 'broken' shift
        pencil_top = Line(ORIGIN, DOWN * pencil_length/2, color="#FFFF00", stroke_width=10)
        pencil_bottom = Line(DOWN * pencil_length/2, DOWN * pencil_length, color="#FFFF00", stroke_width=10)
        pencil = VGroup(pencil_top, pencil_bottom).rotate(pencil_angle)
        
        # Fix Issue 30: Start pencil at grid B4
        self.place_at_grid(pencil, 'B4', scale_factor=0.7)
        
        # Animate pencil being dipped into the water
        target_pos = self.grid['D4']
        self.play(pencil.animate.move_to(target_pos), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#ADD8E6") # Medium Blue for refraction effect
        
        # Horizontally shift the submerged portion of the pencil
        self.play(pencil_bottom.animate.shift(RIGHT * 0.4), run_time=1)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Highlight Line 4
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFD700") # Gold for coin
        
        coin = Circle(radius=0.2, color="#FFD700", fill_opacity=1)
        # Fix Issue 31: Place coin at the bottom of the glass (F4)
        self.place_at_grid(coin, 'F4', scale_factor=0.5)
        
        self.play(FadeIn(coin))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight Line 5
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFD700")
        
        # Define light ray points
        surface_y = water.get_top()[1]
        coin_pos = coin.get_center()
        surface_point = np.array([coin_pos[0] + 0.5, surface_y, 0])
        
        # Incident ray
        ray_incident = Line(coin_pos, surface_point, color="#FFD700")
        # Normal line
        normal = DashedLine(surface_point + UP * 0.6, surface_point + DOWN * 0.6, color=WHITE)
        # Refracted ray
        refracted_end = surface_point + np.array([1.5, 0.9, 0])
        ray_refracted = Line(surface_point, refracted_end, color="#FFD700")
        
        # Apparent position and ray
        apparent_coin_pos = coin_pos + UP * 0.8 + RIGHT * 0.15
        apparent_ray = DashedLine(refracted_end, apparent_coin_pos, color="#FFD700", stroke_opacity=0.4)
        apparent_coin = Circle(radius=0.2, color="#FFD700", fill_opacity=0.3).move_to(apparent_coin_pos)

        self.play(Create(ray_incident))
        self.play(Create(normal))
        self.play(Create(ray_refracted))
        self.play(Create(apparent_ray), FadeIn(apparent_coin))
        
        self.wait(3)
